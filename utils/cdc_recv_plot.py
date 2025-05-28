#!/usr/bin/env python3
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

PORT = '/dev/ttyACM0'
BAUDRATE = 115200 * 32  # 其实对 CDC 来说随便写个大数
BLOCK = 65536  # 每次最多读 64 k
TARGET_SZ = 1024 * 1024  # 1 MiB


def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0)
    except serial.SerialException as e:
        sys.exit(f'打开串口失败: {e}')

    print(f'Listening {PORT} … 目标 {TARGET_SZ // 1024} KB')
    t0 = time.time()
    cnt_speed = 0
    buf = bytearray()

    while len(buf) < TARGET_SZ:
        chunk = ser.read(BLOCK)
        n = len(chunk)
        if n:
            buf.extend(chunk)
            cnt_speed += n

        # 每秒打印一次速率
        if time.time() - t0 >= 1.0:
            print(f'Speed: {cnt_speed / 1024 / 1024:.2f} MB/s | '
                  f'已获取 {len(buf) * 100 / TARGET_SZ:.1f}%')
            cnt_speed = 0
            t0 = time.time()

    ser.close()
    print('采集完成！总字节数 =', len(buf))

    # ---------- 绘图 ----------
    data = np.frombuffer(buf, dtype=np.uint8)  # 或根据实际协议换 dtype
    x = np.arange(data.size)

    plt.figure(figsize=(10, 4))
    plt.plot(x, data, lw=0.5)
    plt.title('CDC-ACM capture (1 MiB)')
    plt.xlabel('Sample index')
    plt.ylabel('Value (uint8)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
