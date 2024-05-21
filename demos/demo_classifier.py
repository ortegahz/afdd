import argparse
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

from cores.classifier import ClassifierXGB
from utils.utils import set_logging, svm_label2data, feature_engineering


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_label', default='/home/manu/tmp/afd_v0')
    parser.add_argument('--path_save', default='/home/manu/tmp/model.pickle')
    return parser.parse_args()


def run(args):
    logging.info(args)
    X, y = svm_label2data(args.path_label)
    y[y < 0] = 0
    X = feature_engineering(X)
    # iris = load_iris()
    # X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_names = [str(i) if i < 256 else 'f' + str(i) for i in range(X.shape[1])]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        # 'objective': 'multi:softmax',
        # 'num_class': 3,
        'scale_pos_weight': scale_pos_weight,
    }
    logging.info(f'params -> {params}')
    classifier = ClassifierXGB(params)
    classifier.train(dtrain)
    with open(args.path_save, 'wb') as f:
        pickle.dump(classifier, f)
    with open(args.path_save, 'rb') as f:
        classifier = pickle.load(f)
    preds = classifier.infer(dtest)
    preds = [1 if prob > 0.5 else 0 for prob in preds]
    accuracy = accuracy_score(y_test, preds)
    logging.info(f'acc -> {accuracy}')
    plot_importance(classifier.model)
    plt.show()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
