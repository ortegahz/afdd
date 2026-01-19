import argparse
import logging
import os.path
from collections import Counter

import h5py
import torch
from sklearn.model_selection import train_test_split

from utils import set_logging, make_dirs, load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_label_train', default='/home/Huangzhe/test/afd_pm_train')
    # parser.add_argument('--save_dir', default='/dev/shm/afd_pm_hdf5')
    parser.add_argument('--save_dir', default='/home/Huangzhe/test/afd_pm_hdf5')
    return parser.parse_args()


def svm2hdf5(args):
    logging.info(args)
    make_dirs(args.save_dir, reset=True)
    _seed = 64
    torch.manual_seed(_seed)
    x, y, x_aug, y_aug = load_data(args.path_label_train, lidx=4096 * -4)
    x_train_aug, x_test_aug, y_train_aug, y_test_aug = \
        train_test_split(x_aug, y_aug, test_size=0.3, random_state=_seed, stratify=y_aug)
    # ros = RandomOverSampler(sampling_strategy='auto')
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    logging.info(f'Counter(y_train_aug) -> {Counter(y_train_aug)}')
    logging.info(f'Counter(y_test_aug) -> {Counter(y_test_aug)}')

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=_seed, stratify=y)
    # ros = RandomOverSampler(sampling_strategy='auto')
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    logging.info(f'Counter(y_train) -> {Counter(y_train)}')
    logging.info(f'Counter(y_test) -> {Counter(y_test)}')

    # Define HDF5 file paths
    train_hdf5_path = os.path.join(args.save_dir, 'train_data.h5')
    test_hdf5_path = os.path.join(args.save_dir, 'test_data.h5')

    # Save training data in HDF5 format
    with h5py.File(train_hdf5_path, 'w') as f:
        # f.create_dataset('features', data=x_train_aug)
        # f.create_dataset('labels', data=y_train_aug)
        f.create_dataset('features', data=x_train)
        f.create_dataset('labels', data=y_train)

    # Save testing data in HDF5 format
    with h5py.File(test_hdf5_path, 'w') as f:
        f.create_dataset('features', data=x_test)
        f.create_dataset('labels', data=y_test)


def main():
    set_logging()
    args = parse_args()
    svm2hdf5(args)


if __name__ == '__main__':
    main()
