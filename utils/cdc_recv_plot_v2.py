#!/usr/bin/env python3
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

N_BIT_VALID = 12
WORD_BYTES = 4  # 32-bit = 4 byte
PAIR_BYTES = WORD_BYTES * 2  # (U,I) 一共 8 byte

PORT = '/dev/ttyACM1'
BAUDRATE = 115200 * 32
BLOCK = 65536

# 需要多少原始字节自己算一下；下面示例：抓 1 Mi 采样对
TARGET_PAIR = 1 * 1024 * 1024  # 1 Mi (U,I) 对
TARGET_SZ = TARGET_PAIR * PAIR_BYTES


def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0)
    except serial.SerialException as e:
        sys.exit(f'打开串口失败: {e}')

    print(f'Listening {PORT} … 目标 {TARGET_SZ // 1024} KB')
    t0 = time.time()
    speed_cnt = 0
    buf = bytearray()

    while len(buf) < TARGET_SZ:
        chunk = ser.read(BLOCK)
        if chunk:
            buf.extend(chunk)
            speed_cnt += len(chunk)

        if time.time() - t0 >= 1.0:
            print(f'Speed: {speed_cnt / 1024 / 1024:5.2f} MB/s | '
                  f'已获取 {len(buf) * 100 / TARGET_SZ:5.1f}%')
            speed_cnt = 0
            t0 = time.time()

    ser.close()

    # 裁掉尾部不足一个 (U,I) 对的数据
    full_pairs = len(buf) // PAIR_BYTES
    buf = memoryview(buf)[: full_pairs * PAIR_BYTES]
    print('采集完成！总字节数 =', len(buf))

    # ---------- 解析 ----------
    raw32 = np.frombuffer(buf, dtype='<u4')  # <u4 = little-endian uint32
    voltage = raw32[1::2]  # U0, U1, ...
    current = raw32[0::2]  # I0, I1, ...

    voltage = voltage[:int(1000000 / 50 / 2)]

    # 随便画一个通道做演示
    x = np.arange(voltage.size)
    plt.figure(figsize=(10, 4))
    plt.plot(x, voltage, lw=0.4)
    plt.title('Voltage channel (uint32)')
    plt.xlabel('Sample index')
    plt.ylabel('ADC raw code')
    plt.ylim(0, 2 ** N_BIT_VALID)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
