import argparse

from cores.detector_wrapper import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/负载抑制性试验/电容滤波')
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/demo_arc_detector_save')
    return parser.parse_args()


def run(args):
    detector_wrapper = DetectorWrapperV0(args.addr, args.dir_plot_save)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
