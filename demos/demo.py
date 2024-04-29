import argparse

from cores.detector_wrapper import *
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/故障电弧实验')
    return parser.parse_args()


def run(args):
    detector_wrapper = DetectorWrapperV1(args.addr)
    detector_wrapper.run()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
