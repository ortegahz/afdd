# toolchain
export PATH=/home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/bin:$PATH
export PATH=/home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/libexec/gcc/arm-rockchip830-linux-uclibcgnueabihf/8.3.0:$PATH

chmod a+x /home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-*
chmod a+x /home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/libexec/gcc/arm-rockchip830-linux-uclibcgnueabihf/8.3.0/cc1

TC=/home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/bin
cd  $TC
ln -sf arm-rockchip830-linux-uclibcgnueabihf-as  as

which as
as --version

# device
telnet 172.20.20.144
root
0

# nfs
mount -t nfs 172.20.20.27:/home/manu/nfs /mnt_manu -o nolock

# run
insmod ./timer.ko
rmmod timer