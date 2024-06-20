import argparse
import logging

from cores.pico import Pico5444DMSO
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def run(args):
    logging.info(args)
    pico = Pico5444DMSO()
    pico.sample()
    pico.plot()
    pico.close()
    logging.info(pico)


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
