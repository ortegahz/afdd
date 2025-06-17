#!/usr/bin/env python3
"""
High-speed USB-CDC 捕获示例（优化版）
author: you
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

# ---------------- 用户可调常量 ----------------
# PAIR_BYTES = 4  # 16-bit V + 16-bit I
# LEN = 1000 * 1000
PAIR_BYTES = 8  # 16-bit V * 4
LEN = 20 * 1000
PORT = '/dev/ttyACM0'
BAUDRATE = 115200 * 32  # 板端 CDC-ACM 固件的实际速率
USE_RTSCTS = False  # 如固件支持，可改 True
BLOCK = 128 * 1024  # 每次 readinto 尝试读取字节数
TARGET_SZ = PAIR_BYTES * LEN  # 目标采样原始字节数
N_BIT_VALID = 12  # 有效 ADC 位宽


# ------------------------------------------------

def main() -> None:
    try:
        ser = serial.Serial(
            PORT,
            BAUDRATE,
            timeout=0.05,  # 阻塞，最长 50 ms 超时
            rtscts=USE_RTSCTS
        )
    except serial.SerialException as e:
        sys.exit(f'串口打开失败: {e}')

    print(f'Listening on {PORT} … 目标 {TARGET_SZ // 1024} KB')
    t0 = time.time()
    speed_byte = 0  # 1 s 内已读字节数
    pos = 0  # 已写入 buffer 的游标
    buf = bytearray(TARGET_SZ)  # 预分配整块内存
    mv = memoryview(buf)  # 避免每次重新包装

    while pos < TARGET_SZ:
        # 剩余可写空间
        room = TARGET_SZ - pos
        n_max = BLOCK if room >= BLOCK else room

        # 直接把数据读到预分配缓冲
        n_read = ser.readinto(mv[pos:pos + n_max])
        if n_read:
            pos += n_read
            speed_byte += n_read

        # 每秒打印一次速率
        if time.time() - t0 >= 1.0:
            print(f'Speed: {speed_byte / 1024 / 1024:5.2f} MB/s | '
                  f'完成 {pos * 100 / TARGET_SZ:5.1f}%')
            speed_byte = 0
            t0 = time.time()

    ser.close()
    print('采集完成！总字节数 =', pos)

    # ---------------- 解析并绘图 ----------------
    full_pairs = pos // PAIR_BYTES
    mv_pairs = mv[:full_pairs * PAIR_BYTES]

    raw16 = np.frombuffer(mv_pairs, dtype='<u2')
    voltage = raw16[1::int(PAIR_BYTES / 2)]  # V 在偶数索引 or 奇数索引请按实际调整
    # current = raw16[::2]

    # 只画前 1 Msamples（可视化需要）
    n_plot = voltage.size
    # n_plot = int(LEN / 50)
    x = np.arange(n_plot)

    plt.figure(figsize=(10, 4))
    plt.plot(x, voltage[:n_plot], lw=0.4)
    plt.title('Voltage channel (uint16), 1 MiB capture')
    plt.xlabel('Sample index')
    plt.ylabel('ADC raw code')
    plt.ylim(0, 2 ** N_BIT_VALID)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
