import argparse
import logging

from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def run(args):
    logging.info(args)


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
