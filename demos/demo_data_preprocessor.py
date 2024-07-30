import argparse

from utils.utils import set_logging
from cores.preprocessor_wrapper import *


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v2/org/0627/并联碳化路径试验/并联碳化路径3-1-报警-AF01.csv')
    # parser.add_argument('--dtr_type', default='PreprocessorWrapperV0')
    # parser.add_argument('--dir_plot_save', default='/home/manu/tmp/demo_afdd_save')
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v2/org/')
    parser.add_argument('--dtr_type', default='PreprocessorWrapperV1')
    parser.add_argument('--dir_plot_save', default='/media/manu/data/afdd/data/data_v2/subsample_v1/')
    parser.add_argument('--path_label', default='/media/manu/data/afdd/data/data_v2/org/波形采样率统计.xlsx')
    return parser.parse_args()


def run(args):
    detector_wrapper = eval(args.dtr_type)(args.addr, args.dir_plot_save, args.path_label)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
