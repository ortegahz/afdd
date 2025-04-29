ARCH := arm
CROSS_COMPILE := arm-rockchip830-linux-uclibcgnueabihf-
ccflags-y := -std=gnu11 -Wno-declaration-after-statement
KERNELDIR := /home/manu/sdk/RK1103/01-SDK/Linux/sysdrv/source/kernel
CURRENT_PATH := $(shell pwd)

obj-m := timer.o

build: kernel_modules

kernel_modules:
	$(MAKE) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) -C $(KERNELDIR) M=$(CURRENT_PATH) modules
	rm *.mod*
	cp timer.ko /home/manu/nfs/out/ -rvf
clean:
	$(MAKE) -C $(KERNELDIR) M=$(CURRENT_PATH) clean
