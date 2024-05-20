import argparse
import logging

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cores.classifier import ClassifierXGB
from utils.utils import set_logging, svm_label2data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_label', default='/home/manu/tmp/smartsd')
    return parser.parse_args()


def run(args):
    logging.info(args)
    X, y = svm_label2data(args.path_label)
    y[y < 0] = 0
    # iris = load_iris()
    # X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    classifier = ClassifierXGB()
    classifier.train(dtrain)
    preds = classifier.infer(dtest)
    preds = [1 if prob > 0.5 else 0 for prob in preds]
    accuracy = accuracy_score(y_test, preds)
    logging.info(f'acc -> {accuracy}')


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
