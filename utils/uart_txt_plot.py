import matplotlib.pyplot as plt
import numpy as np

N_BIT_VALID = 12  # 有效 ADC 位宽

# 1) 读十六进制文件 ---------------------------------------------------
file_name = '/home/manu/tmp/Serial Debug 2025-05-26 111240.txt'

with open(file_name, 'r') as f:
    hex_str_list = [line.strip() for line in f if line.strip()]

u16 = np.array([int(x, 16) for x in hex_str_list], dtype=np.uint16)  # 无符号
i16 = u16.view(np.int16)  # 有符号

# 2) 画图 --------------------------------------------------------------
plt.figure(figsize=(9, 4))

# 如果想画无符号
plt.plot(u16, marker='o', lw=1, label='uint16')

# 如果想同时比较有符号，也可以再画一条
plt.plot(i16, marker='x', lw=1, label='int16 (same raw bits)')

plt.title('Hex data from file: {}'.format(file_name.split('/')[-1]))
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.grid(True)
plt.ylim(0, 2 ** N_BIT_VALID)
plt.legend()
plt.tight_layout()
plt.show()
