import argparse

from utils.utils import set_logging
from cores.preprocessor_wrapper import *


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v2/org/试验数据0716/误报警试验/电暖多次启停.csv')
    # parser.add_argument('--dtr_type', default='PreprocessorWrapperV0')
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v2/org/试验数据0716')
    parser.add_argument('--dtr_type', default='PreprocessorWrapperV1')
    parser.add_argument('--dir_plot_save', default='/media/manu/data/afdd/data/data_v2/subsample/试验数据0716')
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
