import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 文件路径
file_path1 = '/home/manu/tmp/seq_state_pred_idle.txt'
root, ext = os.path.splitext(file_path1)
file_path2 = f'{root}_cpp{ext}'

# 读取数据
data1 = np.loadtxt(file_path1)
data2 = np.loadtxt(file_path2)

# 计算多种相似度指标
cosine_similarity = 1 - cosine(data1, data2)
mse = mean_squared_error(data1, data2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data1, data2)
correlation = np.corrcoef(data1, data2)[0, 1]

# 计算相对误差
relative_error = np.mean(np.abs(data1 - data2) / (np.abs(data1) + 1e-10)) * 100

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 第一个子图：显示两个序列
ax1.plot(data1, label='s1', alpha=0.7)
ax1.plot(data2, label='s2', linestyle='--', alpha=0.7)
ax1.set_xlabel('Index')
ax1.set_ylabel('Value')
ax1.set_title('Comparison of Sequence States')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 第二个子图：显示差异
difference = data1 - data2
ax2.plot(difference, label='Difference (data1 - data2)', color='red')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Index')
ax2.set_ylabel('Difference')
ax2.set_title('Difference between Sequences')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 在图上显示多个指标
metrics_text = (f"Cosine Similarity: {cosine_similarity:.4f}\n"
                f"Correlation: {correlation:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"Relative Error: {relative_error:.2f}%")

ax1.text(0.02, 0.98, metrics_text,
         fontsize=10, transform=ax1.transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# 打印详细统计信息
print("=== 详细统计信息 ===")
print(f"数据长度: {len(data1)}")
print(f"\ndata1 统计:")
print(f"  均值: {np.mean(data1):.4f}")
print(f"  标准差: {np.std(data1):.4f}")
print(f"  最小值: {np.min(data1):.4f}")
print(f"  最大值: {np.max(data1):.4f}")
print(f"\ndata2 统计:")
print(f"  均值: {np.mean(data2):.4f}")
print(f"  标准差: {np.std(data2):.4f}")
print(f"  最小值: {np.min(data2):.4f}")
print(f"  最大值: {np.max(data2):.4f}")
print(f"\n相似度指标:")
print(f"  余弦相似度: {cosine_similarity:.6f}")
print(f"  皮尔逊相关系数: {correlation:.6f}")
print(f"  均方根误差 (RMSE): {rmse:.6f}")
print(f"  平均绝对误差 (MAE): {mae:.6f}")
print(f"  相对误差: {relative_error:.2f}%")
