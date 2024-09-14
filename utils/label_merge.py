import argparse
import logging
import os

from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_in',
                        default=['/media/manu/ST8000DM004-2U91/afdd/data/data_v1/s400/afd_data_v1_s400',
                                 '/media/manu/ST8000DM004-2U91/afdd/data/data_v2/subsample_v1 - wo pre-filter/afd_data_v2_wof',
                                 '/media/manu/ST8000DM004-2U91/afdd/data/data_v3/demo测试数据/afd_data_v3',
                                 '/media/manu/ST8000DM004-2U91/afdd/data/data_v4/故障电弧测试数据-0904/afd_data_v4',])
    parser.add_argument('--path_out', default='/home/manu/tmp/afd_pm')
    return parser.parse_args()


def run(args):
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


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
