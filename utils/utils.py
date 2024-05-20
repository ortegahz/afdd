import logging
import os
import shutil

import numpy as np


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


def svm_label2data(path_label):
    with open(path_label, 'r') as file:
        lines = file.readlines()
    x, y = list(), list()
    for line in lines:
        line_lst = line.strip().split(' ')
        logging.info(line_lst)
        y.append(int(line_lst[0]))
        x.append([float(item.split(':')[1]) for item in line_lst[1:]])
    return np.array(x).astype(np.float32), np.array(y).astype(np.int64)
