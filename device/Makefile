ARCH            := arm
CROSS_COMPILE   := arm-rockchip830-linux-uclibcgnueabihf-

# 内核路径
KERNELDIR       := /home/manu/sdk/RK1103/01-SDK/Linux/sysdrv/source/kernel
CURRENT_PATH    := $(shell pwd)

SYSROOT         := /home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/arm-rockchip830-linux-uclibcgnueabihf/sysroot/

# ---------------- 内核模块 ----------------
obj-m           := timer.o
ccflags-y       := -std=gnu11 -Wno-declaration-after-statement

# ---------------- 用户程序 ----------------
USER_SRC        := ad738x_reader.c
USER_TARGET     := ad738x_reader     # 生成的可执行文件名

# 输出目录
OUTDIR          := /home/manu/nfs/out

# -------------------------------------------------
.PHONY: all kernel_modules user_program clean
# -------------------------------------------------

all: kernel_modules user_program

# 1) 交叉编译内核模块
kernel_modules:
	$(MAKE) -C $(KERNELDIR) \
			M=$(CURRENT_PATH) \
			ARCH=$(ARCH) \
			CROSS_COMPILE=$(CROSS_COMPILE) \
			modules
	rm -f *.mod.* *.o.cmd
	@mkdir -p $(OUTDIR)
	cp -v timer.ko $(OUTDIR)/

# 2) 交叉编译用户空间工具
user_program: $(USER_SRC)
	@echo "  CC      $(USER_TARGET)"
	$(CROSS_COMPILE)gcc --sysroot=$(SYSROOT) -std=gnu11 -O2 -Wall -o $(USER_TARGET) $(USER_SRC)
	cp -v $(USER_TARGET) $(OUTDIR)/

# 3) 清理
clean:
	$(MAKE) -C $(KERNELDIR) \
			M=$(CURRENT_PATH) \
			ARCH=$(ARCH) \
			CROSS_COMPILE=$(CROSS_COMPILE) \
			clean
	$(RM) $(USER_TARGET)