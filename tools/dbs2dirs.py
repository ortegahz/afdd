import argparse
import logging
import os

from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_in',
                        default=[
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/自主正例 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/自主负例 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/误动作 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/负载抑制 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/串联碳化 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/并联碳化 - o/afd',
                            '/media/manu/ST8000DM004-2U91/afdd/data/data_v26 - duke/data_sorted - o/并联金属 - o/afd',
                        ])
    parser.add_argument('--path_out', default='/media/manu/ST8000DM004-2U91/tmp/afd_dir_list.txt')
    return parser.parse_args()


def run_get_dirs(args):
    """
    提取 paths_in 中每个路径的父目录，并写入 path_out 指定的 txt 文件中。
    """
    if os.path.exists(args.path_out):
        os.remove(args.path_out)

    with open(args.path_out, 'w') as f:
        for path_in in args.paths_in:
            # 获取父目录路径
            dir_path = os.path.dirname(path_in)
            # 确保路径以分隔符结尾
            if not dir_path.endswith(os.sep):
                dir_path += os.sep
            f.write(dir_path + '\n')

    logging.info(f'Directory list saved to {args.path_out}')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run_get_dirs(args)


if __name__ == '__main__':
    main()
