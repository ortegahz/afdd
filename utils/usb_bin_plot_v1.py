import os
import struct

import matplotlib.pyplot as plt
import numpy as np

# 二进制文件路径
FILENAME = "/media/manu/ST8000DM004-2U91/afdd/data/data_v30/data_sorted/串联碳化/正例-串联碳化-额定+1_20k.bin"
N_BIT_VALID = 12  # ADC 有效位数


def parse_and_plot_bin_file():
    """解析二进制文件并将两个通道波形画在同一张图里"""
    print(f"开始解析文件: {FILENAME}")

    if not os.path.exists(FILENAME):
        print(f"错误: 文件 {FILENAME} 不存在")
        return

    file_size = os.path.getsize(FILENAME)
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")

    odd_values, even_values = [], []

    # 读取并解析
    with open(FILENAME, 'rb') as f:
        data = f.read()
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                value1 = struct.unpack('<H', data[i:i + 2])[0]  # 奇数通道
                value2 = struct.unpack('<H', data[i + 2:i + 4])[0]  # 偶数通道
                odd_values.append(value1)
                even_values.append(value2)

    print(f"解析完成: 共 {len(odd_values)} 组数据")

    odd_array = np.asarray(odd_values, dtype=np.uint16)  # ARC
    even_array = np.asarray(even_values, dtype=np.uint16)  # UAC

    print(f"ARC: {np.min(odd_array)} - {np.max(odd_array)}")
    print(f"UAC: {np.min(even_array)} - {np.max(even_array)}")

    # ==============  绘图 =================
    plt.figure(figsize=(12, 6))
    plt.plot(even_array, 'r-', label='UAC (V)')
    plt.plot(odd_array, 'b-', label='ARC (A)')

    plt.title(f'文件解析结果: {os.path.basename(FILENAME)}')
    plt.xlabel('point')
    plt.ylabel('ADC Value')
    plt.ylim(0, 2 ** N_BIT_VALID)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parse_and_plot_bin_file()
