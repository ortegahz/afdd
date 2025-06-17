import os
import struct

import matplotlib.pyplot as plt
import numpy as np

# 指定要解析的文件名
FILENAME = "/media/manu/ST8000DM004-2U91/afdd/data/data_v30/6.11故障电弧测试数据/自主设计/探测器和电弧发生装置距离200米/反例-多负载运行（冰箱+电磁炉+日光灯+电钻+吸尘器最大转速），随机启停一种负载.bin"
# FILENAME = "/home/manu/tmp/bins_out/国标/国标正例串联碳化-4+1_20k.bin"
N_BIT_VALID = 12


def parse_and_plot_bin_file():
    """解析二进制文件并绘制图形"""
    print(f"开始解析文件: {FILENAME}")

    # 检查文件是否存在
    if not os.path.exists(FILENAME):
        print(f"错误: 文件 {FILENAME} 不存在")
        return

    # 获取文件大小
    file_size = os.path.getsize(FILENAME)
    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")

    # 创建数组存储解析后的数据
    even_values = []
    odd_values = []

    # 读取并解析文件
    with open(FILENAME, 'rb') as f:
        # 读取整个文件内容
        data = f.read()

        # 处理每组4字节数据
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                # 解析第一个uint16值 (前两个字节)
                value1 = struct.unpack('<H', data[i:i + 2])[0]
                # 解析第二个uint16值 (后两个字节)
                value2 = struct.unpack('<H', data[i + 2:i + 4])[0]

                # 第一个值存入odd_values (奇数位置)
                odd_values.append(value1)
                # 第二个值存入even_values (偶数位置)
                even_values.append(value2)

        # # 处理每组4字节数据
        # for i in range(0, len(data), 2):
        #     if i + 3 < len(data):
        #         # 解析第一个uint16值 (前两个字节)
        #         value1 = struct.unpack('<H', data[i:i + 2])[0]
        #
        #         # 第一个值存入odd_values (奇数位置)
        #         odd_values.append(value1)
        #         # 第二个值存入even_values (偶数位置)
        #         even_values.append(0)

    print(f"解析完成: 共 {len(odd_values)} 组数据")

    # 转换为numpy数组
    odd_array = np.array(odd_values, dtype=np.uint16)
    even_array = np.array(even_values, dtype=np.uint16)

    print(f"ARC: {np.min(odd_array)} - {np.max(odd_array)}")
    print(f"UAC: {np.min(even_array)} - {np.max(even_array)}")

    # _s_idx, _e_idx = 0, 1024 * 1024
    # odd_array = odd_array[_s_idx:_e_idx]
    # even_array = even_array[_s_idx:_e_idx]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制偶数位置数据
    plt.subplot(2, 1, 1)
    plt.plot(even_array, 'r-')
    plt.title('UAC (V)')
    plt.xlabel('point')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2 ** N_BIT_VALID)

    # 绘制奇数位置数据
    plt.subplot(2, 1, 2)
    plt.plot(odd_array, 'b-')
    plt.title('ARC (A)')
    plt.xlabel('point')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2 ** N_BIT_VALID)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.suptitle(f'文件解析结果: {FILENAME}', fontsize=16, y=1.02)
    plt.show()


# 执行解析和绘图
if __name__ == "__main__":
    parse_and_plot_bin_file()
