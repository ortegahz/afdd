#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIR_BYTES = 4  # 2B V + 2B I
N_BIT_VALID = 12  # ADC 有效位宽
RAW_FILE = '/home/manu/tmp/raw.bin'  # dd 得到的文件
N_BIT_VALID = 12  # 有效 ADC 位宽


def main():
    raw = Path(RAW_FILE).read_bytes()
    full_pairs = len(raw) // PAIR_BYTES
    if len(raw) % PAIR_BYTES:
        print('Warning: 文件最后留有残余字节，已自动丢弃')
        raw = raw[:full_pairs * PAIR_BYTES]

    # 以 uint16 little-endian 解码
    u16 = np.frombuffer(raw, dtype='<u2')  # < 表示 little-endian
    current = u16[::2]  # 偶数索引 I
    voltage = u16[1::2]  # 奇数索引 V
    print(f'一共解析出 {voltage.size} 组 (V,I) 样本')

    # 只画前 1e5 点
    n = max(100_000, voltage.size)
    plt.figure(figsize=(10, 4))
    plt.plot(voltage[:n], lw=0.4)
    plt.ylim(0, 2 ** N_BIT_VALID)
    plt.title('Voltage channel (mV)')
    plt.xlabel('Sample index')
    plt.ylabel('mV')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
