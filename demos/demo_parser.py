import argparse

from parsers.parser import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v1s_s400/故障电弧试验/串联碳化路径4kw-1-无电弧.npy')
    parser.add_argument('--db_type', default='DataV3')
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/afdd_demo_parser_save')
    return parser.parse_args()


def run(args):
    logging.info(args)
    parser = ParserV0(db_type=args.db_type, addr_in=args.addr, dir_plot_save=args.dir_plot_save)
    parser.parse()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
