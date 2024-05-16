import argparse

from parsers.parser import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/home/manu/tmp/并联碳化3-1-故障电弧14个-削波.csv')
    parser.add_argument('--db_type', default='DataV1')
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
