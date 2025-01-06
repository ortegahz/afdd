import argparse
import logging
import os

import torch

from cores.classifier import ClassifierCNN
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default='/home/manu/tmp/afd_pm_hdf5')
    parser.add_argument('--path_ckpt', default='/home/manu/tmp/afdd_models/best_e68.pt')
    return parser.parse_args()


def run_cnn(args):
    logging.info(args)
    _seed = 128
    torch.manual_seed(_seed)
    classifier = ClassifierCNN(args=args.path_ckpt, is_infer=True)
    _data = {
        'train_path': os.path.join(args.load_dir, 'train_data.h5'),
        'test_path': os.path.join(args.load_dir, 'test_data.h5'),
    }
    val_accuracy = classifier.evaluate(_data['test_path'])
    logging.info(f'best val_accuracy -> {val_accuracy}')


def main():
    set_logging()
    args = parse_args()
    run_cnn(args)


if __name__ == '__main__':
    main()
