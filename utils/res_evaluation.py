import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

# 文件路径
file_path1 = '/home/manu/tmp/seq_state_pred_classifier.txt'
file_path2 = '/home/manu/tmp/062F8DF8-0198-4465-80CF-1C6931BECF24.txt'

# 读取数据
data1 = np.loadtxt(file_path1)
# data2 = np.loadtxt(file_path2)
data2 = np.genfromtxt(file_path2, delimiter=",", filling_values=0, encoding='utf-16')

# data3 = np.ones_like(data2) * 2048
# data2 = data3

# 计算余弦相似度
# cosine_similarity = 1 - cosine(data1, data2)

# 绘图
# plt.plot(data1, label='Sequence State file_path1')
plt.plot(data2, label='Sequence State file_path2', linestyle='--')

# # 在图上显示余弦相似度
# plt.text(0.5, 0.5, f"Cosine Similarity: {cosine_similarity:.4f}",
#          fontsize=12, transform=plt.gca().transAxes,
#          verticalalignment='center', horizontalalignment='center')

# 添加标签和标题
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Sequence State')
plt.legend()
plt.grid(True)
plt.show()
