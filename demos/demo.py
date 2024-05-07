import argparse

from cores.detector_wrapper import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/误动作实验/吸尘器')
    parser.add_argument('--dtr_type', default='DetectorWrapperV0')
    parser.add_argument('--db_key', default='20211022_1')
    # parser.add_argument('--addr', default='/media/manu/data/afdd/误动作实验')
    # parser.add_argument('--dtr_type', default='DetectorWrapperV1')
    # parser.add_argument('--db_key', default=None)
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/demo_arc_detector_save')
    return parser.parse_args()


def run(args):
    detector_wrapper = eval(args.dtr_type)(args.addr, args.dir_plot_save, args.db_key)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
