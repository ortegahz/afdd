import argparse
import logging

from data.data import DataV0
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/afdd/故障电弧实验/并联碳化3KW_1')
    return parser.parse_args()


def run(args):
    logging.info(args)
    db_obj = DataV0(args.dir_in)
    db_obj.load()
    db_obj.plot()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
