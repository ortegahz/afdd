// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/gpio.h>
#include <linux/hrtimer.h>
#include <linux/ktime.h>
#include <linux/spi/spi.h>
#include <linux/kfifo.h>
#include <linux/poll.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/atomic.h>
#include <linux/delay.h>

#define TARGET_GPIO 145
#define FIFO_BYTES 4096 /* 字节 */
#define FRAME_BYTES 4	/* 1 帧 = 2×16bit */

struct ad738x_data
{
	struct spi_device *spi;
	struct hrtimer timer;
	ktime_t period;
	atomic_t busy;	   /* SPI 事务是否在进行           */
	atomic_t lost_cnt; /* 丢帧计数 0‑15                */

	/* ----------- 一次异步 message ----------- */
	struct spi_message msg;
	struct spi_transfer xfer[2];
	u16 tx_dummy[2];
	u16 rx[2];

	/* ----------- FIFO 与字符设备 ----------- */
	struct kfifo fifo;
	spinlock_t fifo_lock;
	wait_queue_head_t read_wait;

	struct cdev cdev;
	dev_t devno;
	struct class *dev_class;
	struct device *dev;
};

static int ad738x_reg_write(struct spi_device *spi, u8 reg, u16 val)
{
	u16 w = (1 << 15) | (reg << 12) | (val & 0x0fff);
	return spi_write(spi, &w, sizeof(w));
}

/* ---------------- 字符设备 ---------------- */
static ssize_t ad738x_read(struct file *filp,
						   char __user *buf, size_t cnt, loff_t *ppos)
{
	struct ad738x_data *d = filp->private_data;
	unsigned int copied, avail;
	int ret;

	if (cnt & (FRAME_BYTES - 1))
		return -EINVAL;

retry:
	avail = kfifo_len(&d->fifo);
	avail -= avail % FRAME_BYTES;
	cnt = min_t(size_t, cnt, avail);

	if (!cnt)
	{
		if (filp->f_flags & O_NONBLOCK)
			return -EAGAIN;
		if (wait_event_interruptible(d->read_wait,
									 (avail = kfifo_len(&d->fifo)) >= FRAME_BYTES))
			return -ERESTARTSYS;
		goto retry;
	}

	spin_lock_irq(&d->fifo_lock);
	ret = kfifo_to_user(&d->fifo, buf, cnt, &copied);
	spin_unlock_irq(&d->fifo_lock);

	return ret ? ret : copied;
}

static unsigned int ad738x_poll(struct file *filp, poll_table *wait)
{
	struct ad738x_data *d = filp->private_data;
	unsigned int mask = 0;

	poll_wait(filp, &d->read_wait, wait);
	if (kfifo_len(&d->fifo) >= FRAME_BYTES)
		mask |= POLLIN | POLLRDNORM;
	return mask;
}

static int ad738x_open(struct inode *ino, struct file *filp)
{
	filp->private_data =
		container_of(ino->i_cdev, struct ad738x_data, cdev);
	return 0;
}

static const struct file_operations ad738x_fops = {
	.owner = THIS_MODULE,
	.open = ad738x_open,
	.read = ad738x_read,
	.poll = ad738x_poll,
	.llseek = no_llseek,
};

/* ---------- spi complete：收帧入 FIFO ---------- */
static void ad738x_complete(void *arg)
{
	struct ad738x_data *d = arg;
	u16 samples[2];
	int ret;
	int lost;

	/*
	 * 取出并清零丢帧计数；最大只需要 0‑15 ，超过则饱和成 15。
	 * counter 的高 2 位写入电压样本 LSB，
	 * 低 2 位写入电流样本 LSB。
	 */
	lost = atomic_xchg(&d->lost_cnt, 0);
	if (lost > 15)
		lost = 15;

	samples[0] = (d->rx[0] & ~0x3) | ((lost >> 2) & 0x3); /* 电压 */
	samples[1] = (d->rx[1] & ~0x3) | (lost & 0x3);		  /* 电流 */

	if (lost)
	{
		samples[0] = 0xFFFF;
		samples[1] = 0xFFFF;
	}

	/* 压入环形 FIFO，如满则丢掉最旧一帧 */
	spin_lock(&d->fifo_lock);
	while (kfifo_avail(&d->fifo) < FRAME_BYTES)
	{
		u16 dummy[2];
		ret = kfifo_out(&d->fifo, dummy, FRAME_BYTES);
		(void)ret;
	}
	kfifo_in(&d->fifo, samples, FRAME_BYTES);
	spin_unlock(&d->fifo_lock);

	wake_up_interruptible(&d->read_wait);
	gpio_set_value(TARGET_GPIO, 0);
	atomic_set(&d->busy, 0);
}

/* ---------------- hrtimer ------------------ */
static enum hrtimer_restart timer_cb(struct hrtimer *t)
{
	struct ad738x_data *d = container_of(t, struct ad738x_data, timer);

	if (atomic_read(&d->busy))
	{
		/* SPI 仍在忙，认为这一周期“丢失” */
		if (atomic_read(&d->lost_cnt) < 15)
			atomic_inc(&d->lost_cnt); /* 最多累积到 15           */
		goto out;
	}

	atomic_set(&d->busy, 1);
	gpio_set_value(TARGET_GPIO, 1); /* 触发转换 */

	if (spi_async(d->spi, &d->msg))
	{
		atomic_set(&d->busy, 0);
		gpio_set_value(TARGET_GPIO, 0);
	}

out:
	hrtimer_forward_now(t, d->period);
	return HRTIMER_RESTART;
}

/* ---------------- probe/remove ------------- */
static int ad738x_probe(struct spi_device *spi)
{
	struct ad738x_data *d;
	int ret;

	d = devm_kzalloc(&spi->dev, sizeof(*d), GFP_KERNEL);
	if (!d)
		return -ENOMEM;

	/* GPIO */
	ret = gpio_request(TARGET_GPIO, "ad738x_trig");
	if (ret)
		return ret;
	gpio_direction_output(TARGET_GPIO, 0);

	/* SPI 基本参数 */
	spi->mode = SPI_MODE_2;
	spi->bits_per_word = 16;
	ret = spi_setup(spi);
	if (ret)
		goto err_gpio;

	/* ADC 配置示例 */
	ret = ad738x_reg_write(spi, 0x02, 0x003C);
	if (ret)
		goto err_gpio;
	ret = ad738x_reg_write(spi, 0x01, 0x0402);
	if (ret)
		goto err_gpio;

	/* message：两次 16bit，中间抬 CS */
	spi_message_init(&d->msg);

	d->tx_dummy[0] = 0;
	d->tx_dummy[1] = 0;

	d->xfer[0].tx_buf = &d->tx_dummy[0];
	d->xfer[0].rx_buf = &d->rx[0];
	d->xfer[0].len = 2;
	d->xfer[0].bits_per_word = 16;
	d->xfer[0].cs_change = 1; /* <‑‑ 关键，抬 CS */

	d->xfer[1].tx_buf = &d->tx_dummy[1];
	d->xfer[1].rx_buf = &d->rx[1];
	d->xfer[1].len = 2;
	d->xfer[1].bits_per_word = 16;

	spi_message_add_tail(&d->xfer[0], &d->msg);
	spi_message_add_tail(&d->xfer[1], &d->msg);
	d->msg.complete = ad738x_complete;
	d->msg.context = d;

	/* 其它成员 */
	d->spi = spi;
	d->period = ktime_set(0, 500000); /* ≈22.3 kHz */
	atomic_set(&d->busy, 0);
	atomic_set(&d->lost_cnt, 0);

	if (kfifo_alloc(&d->fifo, FIFO_BYTES, GFP_KERNEL))
	{
		ret = -ENOMEM;
		goto err_gpio;
	}
	spin_lock_init(&d->fifo_lock);
	init_waitqueue_head(&d->read_wait);

	/* 字符设备 */
	ret = alloc_chrdev_region(&d->devno, 0, 1, "ad738x");
	if (ret)
		goto err_fifo;

	cdev_init(&d->cdev, &ad738x_fops);
	ret = cdev_add(&d->cdev, d->devno, 1);
	if (ret)
		goto err_chrdev;

	d->dev_class = class_create(THIS_MODULE, "ad738x");
	if (IS_ERR(d->dev_class))
	{
		ret = PTR_ERR(d->dev_class);
		goto err_cdev;
	}
	d->dev = device_create(d->dev_class, NULL, d->devno, NULL, "ad738x");
	if (IS_ERR(d->dev))
	{
		ret = PTR_ERR(d->dev);
		goto err_class;
	}

	/* hrtimer */
	hrtimer_init(&d->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	d->timer.function = timer_cb;
	hrtimer_start(&d->timer, d->period, HRTIMER_MODE_REL);

	spi_set_drvdata(spi, d);
	return 0;

/* ---- 错误清理 ---- */
err_class:
	class_destroy(d->dev_class);
err_cdev:
	cdev_del(&d->cdev);
err_chrdev:
	unregister_chrdev_region(d->devno, 1);
err_fifo:
	kfifo_free(&d->fifo);
err_gpio:
	gpio_set_value(TARGET_GPIO, 0);
	gpio_free(TARGET_GPIO);
	return ret;
}

static int ad738x_remove(struct spi_device *spi)
{
	struct ad738x_data *d = spi_get_drvdata(spi);

	device_destroy(d->dev_class, d->devno);
	class_destroy(d->dev_class);

	hrtimer_cancel(&d->timer);
	while (atomic_read(&d->busy))
		usleep_range(1000, 2000);

	cdev_del(&d->cdev);
	unregister_chrdev_region(d->devno, 1);
	kfifo_free(&d->fifo);

	gpio_set_value(TARGET_GPIO, 0);
	gpio_free(TARGET_GPIO);
	return 0;
}

static const struct of_device_id ad738x_of_ids[] = {
	{.compatible = "adi,ad738x-hrtimer-sampler"},
	{}};
MODULE_DEVICE_TABLE(of, ad738x_of_ids);

static struct spi_driver ad738x_driver = {
	.driver = {
		.name = "ad738x-hrtimer-async",
		.of_match_table = ad738x_of_ids,
	},
	.probe = ad738x_probe,
	.remove = ad738x_remove,
};
module_spi_driver(ad738x_driver);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("AD738x hrtimer sampler (async, voltage+current, lost‑frame flag)");
MODULE_AUTHOR("Your Name");