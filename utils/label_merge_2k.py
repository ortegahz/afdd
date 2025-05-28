import argparse
import logging
import os

from tqdm import tqdm

from macros import *
from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_in',
                        default=[
                            "/media/manu/ST8000DM004-2U91/afdd/data/data_v25/data_pick/neg/afd",
                            "/media/manu/ST8000DM004-2U91/afdd/data/data_v25/data_pick/pos/并联碳化/afd",
                            "/media/manu/ST8000DM004-2U91/afdd/data/data_v25/data_pick/pos/串联碳化/afd",
                            "/media/manu/ST8000DM004-2U91/afdd/data/data_v25/data_pick/pos/负载抑制/afd",
                            "/media/manu/ST8000DM004-2U91/afdd/data/data_v25/data_pick/pos/自主正例/afd",
                        ])
    parser.add_argument('--path_out', default='/home/manu/tmp/afd_pm_train')
    # parser.add_argument('--path_out', default='/home/manu/tmp/afd_pm_test')
    return parser.parse_args()


def run_v0(args):
    if os.path.exists(args.path_out):
        os.remove(args.path_out)
    npos, nneg = 0, 0
    for path_in in args.paths_in:
        with open(path_in, 'r') as f:
            lines = f.readlines()
        with open(args.path_out, 'a') as f:
            for line in lines:
                label = line.strip().split()[0]
                if '+' in label:
                    npos += 1
                else:
                    nneg += 1
                f.write(line)
    logging.info(f'npos --> {npos} && nneg --> {nneg}')


def run_v1(args):
    _seq_len = int(SAMPLE_RATE // 50)
    if os.path.exists(args.path_out):
        os.remove(args.path_out)
    npos, nneg = 0, 0
    for path_in in tqdm(args.paths_in, desc="Processing files"):
        with open(path_in, 'r') as f:
            lines = f.readlines()
        with open(args.path_out, 'a') as f:
            for line in lines:
                # logging.info(line)
                line_lst = line.strip().split()
                label = line_lst[0]
                if '+' in label:
                    npos += 1
                else:
                    nneg += 1
                _line_new = ''.join(label)
                for i in range(_seq_len):
                    _line_new += ',' + line_lst[i + 1].strip().split(':')[1]
                _line_new += '\r\n'
                # logging.info(_line_new)
                f.write(_line_new)
    logging.info(f'npos --> {npos} && nneg --> {nneg}')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run_v1(args)


if __name__ == '__main__':
    main()
