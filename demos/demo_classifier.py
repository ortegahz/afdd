# FILE: demo_classifier.py

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

from cores.classifier import ClassifierXGB, ClassifierCNN, ClassifierCNNAE
from utils.utils import set_logging, svm_label2data_v1


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_dir', default='/dev/shm/afd_pm_hdf5')
    # parser.add_argument('--save_dir', default='/home/Huangzhe/test/afdd_models_mp')
    parser.add_argument('--load_dir', default='/home/manu/tmp/afd_pm_hdf5')
    parser.add_argument('--save_dir', default='/home/manu/tmp/afdd_models_mp')
    parser.add_argument('--path_save', default='/home/manu/tmp/xgb.pt')
    parser.add_argument('--path_label_train', default='/home/manu/tmp/afd_pm_train')
    # parser.add_argument('--path_label_test', default='/home/Huangzhe/test/afd_pm_test')
    parser.add_argument('--path_ckpt', default=None)
    parser.add_argument('--model_type', type=str, default='cnn-ae', choices=['cnn', 'cnn-ae'],
                        help='Type of CNN model to run: supervised (cnn) or unsupervised AE (cnn-ae)')
    parser.add_argument('--ae_model_type', type=str, default='ae',
                        choices=['ae', 'unet', 'mem-ae', 'unet-mem', 'mem-flow-ae'],
                        help="Type of AutoEncoder architecture to use.")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--qat', default=False, help='enable quant-aware training')
    return parser.parse_args()


def run_xgb(args):
    logging.info(args)
    X, y = svm_label2data_v1(args.path_label_train, lidx=4096 * 32)
    y[y < 0] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos
    # scale_pos_weight = 16
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'scale_pos_weight': scale_pos_weight,
    }
    logging.info(f'params -> {params}')
    classifier = ClassifierXGB(params)
    classifier.train(X_train, y_train)
    with open(args.path_save, 'wb') as f:
        pickle.dump(classifier, f)
    with open(args.path_save, 'rb') as f:
        classifier = pickle.load(f)
    preds = classifier.infer(X_test, y_test)
    preds = [1 if prob > 0.5 else 0 for prob in preds]
    accuracy = accuracy_score(y_test, preds)
    logging.info(f'acc -> {accuracy}')
    plot_importance(classifier.model)
    plt.show()


def _load_data(path_label, lidx=-1):
    x, y = svm_label2data_v1(path_label, lidx)
    y[y < 0] = 0
    # ros = RandomOverSampler(sampling_strategy='auto')
    # x, y = ros.fit_resample(x, y)
    # logging.info(f'ros -> {Counter(y)}')
    num_pos = np.sum(y > 0)
    num_neg = len(y) - num_pos
    alpha = num_neg / (num_pos + num_neg)
    return x, y, alpha


def run_cnn(args, is_distributed):
    logging.info(args)
    _seed = 128
    torch.manual_seed(_seed)
    # x, y, alpha = _load_data(args.path_label_train, lidx=4096 * -4)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=_seed, stratify=y)
    # ros = RandomOverSampler(sampling_strategy='auto')
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    # logging.info(f'Counter(y_train) -> {Counter(y_train)}')
    # logging.info(f'Counter(y_test) -> {Counter(y_test)}')
    # x_train, y_train, alpha = _load_data(args.path_label_train)
    # x_test, y_test, _ = _load_data(args.path_label_test)
    classifier = ClassifierCNN(args, ddp=is_distributed)
    _data = {
        'train_path': os.path.join(args.load_dir, 'train_data.h5'),
        'test_path': os.path.join(args.load_dir, 'test_data.h5'),
    }
    classifier.train(_data)


def run_cnn_ae(args, is_distributed):
    logging.info(args)
    _seed = 128
    torch.manual_seed(_seed)

    classifier = ClassifierCNNAE(args, ddp=is_distributed)
    _data = {
        'train_path': os.path.join(args.load_dir, 'train_data.h5'),
        'test_path': os.path.join(args.load_dir, 'test_data.h5'),
    }
    # The training data in train_path should consist of mostly normal samples.
    # The ClassifierCNNAE will filter out any positive samples during training/evaluation.
    classifier.train(_data)


def main_worker(rank, world_size, args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # torch.cuda.set_device(0)

    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        logging.info(f'Process {rank} of {world_size} is using GPU {args.local_rank}')
    else:
        torch.cuda.set_device(args.local_rank)
        logging.info(f'Running in non-distributed mode on GPU {args.local_rank}')

    set_logging()

    args.rank = rank
    args.world_size = world_size
    if args.model_type == 'cnn':
        run_cnn(args, is_distributed)
    elif args.model_type == 'cnn-ae':
        run_cnn_ae(args, is_distributed)


def main():
    set_logging()
    args = parse_args()

    # run_xgb(args)
    # run_cnn(args)

    # 替换后的新代码
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    args.local_rank = local_rank
    main_worker(rank, world_size, args)


if __name__ == '__main__':
    main()
