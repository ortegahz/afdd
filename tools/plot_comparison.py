import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 准备数据
# ==========================================

df = pd.read_csv('/home/manu/mnt/8gpu_3090/afdd_models_mp_c1/cnn_few_shot_results.txt')

# AE 的分数 (来自你的实验日志)
ae_f1_score = 0.8465

# ==========================================
# 2. 绘图设置
# ==========================================
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-whitegrid')  # 使用更美观的风格

# --- 画 CNN 曲线 (蓝色实线) ---
plt.plot(df['ratio'], df['f1_score'],
         marker='o', markersize=8, linewidth=2.5, color='#1f77b4',
         label='CNN (Supervised)')

# --- 画 AE 基准线 (红色虚线) ---
plt.axhline(y=ae_f1_score, color='#d62728', linestyle='--', linewidth=2.5,
            label=f'AE (Zero-Shot, F1={ae_f1_score:.4f})')

# ==========================================
# 3. 关键区域标注 (Storytelling)
# ==========================================

# A. 标注“死亡谷” (Cold Start Problem)
# 在数据量 < 10% 时，CNN 完全不可用
plt.fill_between(df['ratio'], 0, 1,
                 where=(df['ratio'] <= 0.1),
                 color='gray', alpha=0.15)
plt.text(0.02, 0.4, "DEATH VALLEY\nCNN Fails Completely",
         color='black', fontsize=11, fontweight='bold', ha='left')

# B. 标注“AE 优势区” (AE Advantage Zone)
# 在 CNN 赶上 AE 之前的所有区域，AE 都是更好的选择
# 通过插值找到交叉点 (大约在 0.9 左右)
interp_ratios = np.linspace(0.01, 1.0, 100)
interp_scores = np.interp(interp_ratios, df['ratio'], df['f1_score'])
advantage_mask = interp_scores < ae_f1_score

plt.fill_between(interp_ratios, interp_scores, ae_f1_score,
                 where=advantage_mask,
                 color='#d62728', alpha=0.1, interpolate=True)

plt.text(0.4, ae_f1_score + 0.02, "AE Dominance Zone\n(Better Performance & No Label Cost)",
         color='#d62728', fontsize=11, fontweight='bold', ha='center')

# C. 标注“临界点” (Crossover Point)
# 只有当数据量非常大时，CNN 才反超
plt.annotate('Crossover Point\n(~90% Data Required)',
             xy=(0.92, ae_f1_score), xytext=(0.65, 0.65),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, fontweight='bold')

# ==========================================
# 4. 坐标轴与修饰
# ==========================================
plt.title('The "Data Cost" Reality: AE vs. CNN', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Ratio of Arc Training Samples (Data Collection Cost)', fontsize=13)
plt.ylabel('F1 Score (Performance)', fontsize=13)

# 设置坐标轴范围
plt.ylim(-0.05, 1.05)
plt.xlim(0, 1.05)

# 刻度设置
plt.xticks([0.0, 0.1, 0.2, 0.5, 0.8, 1.0], ['0%', '10%', '20%', '50%', '80%', '100%'])
plt.yticks(np.arange(0, 1.1, 0.1))

plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

# 保存并显示
save_path = '/home/manu/tmp/final_comparison_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")
plt.show()
