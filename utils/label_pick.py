import random
import numpy as np
from tqdm import tqdm


def split_and_save_data(x, y, path_positive, path_negative):
    """ 将正例和负例分别保存到不同的文件中 """
    with open(path_positive, 'w') as pos_file, open(path_negative, 'w') as neg_file:
        for i in tqdm(range(len(y)), desc="Saving original data"):
            line = f"{y[i]}"
            features = " ".join([f"{j}:{x[i][j]}" for j in range(len(x[i]))])
            formatted_line = f"{line} {features}\n"

            if y[i] == 1:
                pos_file.write(formatted_line)
            elif y[i] == -1:
                neg_file.write(formatted_line)


def balance_data(x, y):
    positive_indices = [i for i, label in enumerate(y) if label == 1]
    negative_indices = [i for i, label in enumerate(y) if label == -1]

    sampled_negative_indices = random.sample(negative_indices, len(positive_indices))
    balanced_indices = positive_indices + sampled_negative_indices

    x_balanced = x[balanced_indices]
    y_balanced = y[balanced_indices]

    return x_balanced, y_balanced


def svm_label2data(path_label):
    x, y = list(), list()
    with open(path_label, 'r') as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)
        for line in tqdm(file, total=total_lines, desc="Reading data"):
            line_lst = line.strip().split(' ')
            y.append(int(line_lst[0]))
            x.append([float(item.split(':')[1]) for item in line_lst[1:]])

    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.int64)

    return x, y


def save_balanced_data(path_new_label, x_balanced, y_balanced):
    with open(path_new_label, 'w') as file:
        for i in tqdm(range(len(y_balanced)), desc="Saving balanced data"):
            line = f"{y_balanced[i]}"
            features = " ".join([f"{j}:{x_balanced[i][j]}" for j in range(len(x_balanced[i]))])
            file.write(f"{line} {features}\n")


# 使用示例
path_label = '/home/Huangzhe/test/afd_pm'
path_positive = '/home/Huangzhe/test/afd_pm_pos'
path_negative = '/home/Huangzhe/test/afd_pm_neg'
path_new_label = '/home/Huangzhe/test/afd_pm_pick'

# 读取和分离数据
x, y = svm_label2data(path_label)
split_and_save_data(x, y, path_positive, path_negative)

# 平衡和保存数据
x_balanced, y_balanced = balance_data(x, y)
save_balanced_data(path_new_label, x_balanced, y_balanced)
