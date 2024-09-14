import argparse
import logging
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

from cores.classifier import ClassifierXGB, ClassifierCNN
from utils.utils import set_logging, svm_label2data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_label', default='/home/Huangzhe/Test/manu-pc/tmp/afd_pm')
    parser.add_argument('--path_save', default='/home/Huangzhe/Test/manu-pc/tmp/model.pt')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()


def run_xgb(args):
    logging.info(args)
    X, y = svm_label2data(args.path_label)
    y[y < 0] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos * 0.3
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


def run_cnn(args):
    logging.info(args)
    _seed = 64
    torch.manual_seed(_seed)
    x, y = svm_label2data(args.path_label)
    y[y < 0] = 0
    ros = RandomOverSampler(sampling_strategy='auto')
    x, y = ros.fit_resample(x, y)
    logging.info(f'ros -> {Counter(y)}')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=_seed)
    classifier = ClassifierCNN(args.local_rank, ddp=True)
    classifier.train(x_train, y_train, x_test, y_test, args.path_save)
    if args.local_rank == 0:
        classifier.model.load_state_dict(torch.load(args.path_save, map_location=f'cuda:{args.local_rank}'))
        # with open(args.path_save, 'rb') as f:
        #     classifier = pickle.load(f)
        val_accuracy = classifier.evaluate(x_test, y_test)
        logging.info(f'best val_accuracy -> {val_accuracy}')


def main_worker(rank, world_size, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    set_logging()

    args.rank = rank
    args.local_rank = rank
    logging.info(f'Process {rank} is using GPU {args.local_rank}')
    run_cnn(args)


def main():
    set_logging()
    args = parse_args()
    # run_xgb(args)
    # run_cnn(args)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    main_worker(local_rank, world_size, args)


if __name__ == '__main__':
    main()
