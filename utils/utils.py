import logging
import os
import shutil
from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def make_dirs(dir_root, reset=False):
    if dir_root is None:
        return
    if os.path.exists(dir_root) and reset:
        shutil.rmtree(dir_root)
    os.makedirs(os.path.join(dir_root), exist_ok=True)


def load_bin(path, int_size=2):
    int_list = list()
    with open(path, 'rb') as file:
        while True:
            bytes_read = file.read(int_size)
            if not bytes_read:
                break
            integer = int.from_bytes(bytes_read, 'little')
            int_list.append(integer)
    return int_list


def svm_label2data_v0(path_label):
    with open(path_label, 'r') as file:
        lines = file.readlines()
    x, y = list(), list()
    for i, line in enumerate(lines):
        # if i > 4096 * 4:
        #     break
        line_lst = line.strip().split(' ')
        # logging.info(line_lst)
        y.append(int(line_lst[0]))
        x.append([float(item.split(':')[1]) for item in line_lst[1:]])
    return np.array(x).astype(np.float32), np.array(y).astype(np.int64)


def svm_label2data_v1(path_label, lidx=-1):
    with open(path_label, 'r') as file:
        lines = file.readlines()
    x, y = list(), list()
    for i, line in enumerate(lines):
        if i > lidx > 0:
            break
        line_lst = line.strip().split(',')
        # logging.info(line_lst)
        y.append(int(line_lst[0]))
        x.append([float(item) for item in line_lst[1:]])
    return np.array(x).astype(np.float32), np.array(y).astype(np.int64)


def feature_engineering(data):
    data_array = np.array(data)
    fft_values = np.fft.fft(data_array)
    fft_magnitude = np.abs(fft_values)
    data_new = np.concatenate([data_array, fft_magnitude], axis=1)
    return data_new


def load_data(path_label, lidx=-1):
    x, y = svm_label2data_v1(path_label, lidx)
    y[y < 0] = 0
    ros = RandomOverSampler(sampling_strategy='auto')
    x_aug, y_aug = ros.fit_resample(x, y)
    logging.info(f'ros -> {Counter(y_aug)}')
    # num_pos = np.sum(y_aug > 0)
    # num_neg = len(y_aug) - num_pos
    # alpha = num_neg / (num_pos + num_neg)
    return x, y, x_aug, y_aug
