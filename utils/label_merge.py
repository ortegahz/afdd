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
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v1/s400/afd_data_v1_s400',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v2/subsample_v1 - wo pre-filter/afd_data_v2_wof',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v3/demo测试数据/afd_data_v3',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v4/故障电弧测试数据-0904/afd_data_v4',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v5/故障电弧测试数据-0911/afd_data_v5_new',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v5/data_pick/neg/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v6/故障电弧测试数据-0926/afd_data_v6',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v6/data_pick/并联金属性接触电弧试验/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v6/data_pick/负载抑制性试验/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v7/故障电弧测试数据-11.15/串并联碳化路径试验 - labeled/afd_data_v7',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v7/故障电弧测试数据-11.15/负载抑制性试验1 - labeled/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v7/并联金属性接触电弧试验/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v8/故障电弧测试数据-11.28/afd_data_v8',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v8/hard_case/afd_data_v8_hard',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v8/故障电弧测试数据-11.28/误报警试验 - labeled/反例/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v9/data_pick - labeled/data_pick_neg/afd_data_v9_neg',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v9/data_pick - labeled/data_pick_pos - labeled/afd_data_v9_pos',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v10/data_pick/负载抑制性实验/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v10/data_pick/并联金属性接触/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v11/data_pick/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v12/试验数据2025-1-15/国标试验/并联金属性接触试验/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v12/data_pick_neg/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v12/data_pick_pos/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v13/data_pick/负载抑制实验/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v13/data_pick/误动作实验/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v13/data_pick/自主设计实验正例/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v13/data_pick/自主设计实验负例/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v14/data_pick/neg/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v15/data_pick/neg/afd',
                            '/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v16/data_pick/pos/afd',  # new
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
