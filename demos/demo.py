import argparse

from cores.detector_wrapper import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v0/故障电弧实验/串联碳化4KW_0.7')
    # parser.add_argument('--dtr_type', default='DetectorWrapperV0')
    # parser.add_argument('--db_key', default='20211021_2')
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v0/误动作实验/电容启动式电动机')
    # parser.add_argument('--dtr_type', default='DetectorWrapperV0')
    # parser.add_argument('--db_key', default='20211021_1')
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v0/负载抑制性试验/试验1_定频空调')
    parser.add_argument('--dtr_type', default='DetectorWrapperV0')
    parser.add_argument('--db_key', default='20211029_1')
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v0/负载抑制性试验')
    # parser.add_argument('--dtr_type', default='DetectorWrapperV1')
    # parser.add_argument('--db_key', default=None)
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v0')
    # parser.add_argument('--dtr_type', default='DetectorWrapperV2')
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
