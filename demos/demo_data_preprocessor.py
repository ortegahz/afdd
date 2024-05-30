import argparse

from utils.utils import set_logging
from cores.preprocessor_wrapper import *


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v1s/误报警试验/误报警试验-变频空调-制冷模式启动运行（AF01未报警）.csv')
    # parser.add_argument('--dtr_type', default='PreprocessorWrapperV0')
    parser.add_argument('--addr', default='/media/manu/data/afdd/data/data_v1s/误报警试验')
    parser.add_argument('--dtr_type', default='PreprocessorWrapperV1')
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/demo_data_pp_save')
    return parser.parse_args()


def run(args):
    detector_wrapper = eval(args.dtr_type)(args.addr, args.dir_plot_save)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
