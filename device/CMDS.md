# toolchain
export PATH=/opt/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin:$PATH

chmod a+x /home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-*
chmod a+x /home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/libexec/gcc/arm-rockchip830-linux-uclibcgnueabihf/8.3.0/cc1

TC=/home/manu/softwares/arm-rockchip830-linux-uclibcgnueabihf/bin
cd  $TC
ln -sf arm-rockchip830-linux-uclibcgnueabihf-as  as

which as
as --version

# device
telnet 172.20.20.144
telnet 172.20.20.146
root
0

# nfs
mount -t nfs 172.20.20.27:/home/manu/nfs /mnt_manu -o nolock
mount -t nfs 172.20.20.27:/home/manu/nfs /mnt/nfs_manu/ -o nolock && cd /mnt/nfs_manu/out

# run
insmod timer.ko
rmmod timer