import argparse
import logging
import os
import sys

import h5py
import numpy as np

from utils.macros import SAMPLE_RATE

PERIOD_SAMPLES = int(SAMPLE_RATE / 50)
NUM_PERIODS = 128
TEST_LENGTH = NUM_PERIODS * PERIOD_SAMPLES


def set_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将 afd.h5 文件分割为训练集和测试集。")
    parser.add_argument('--input_h5', type=str, default='/media/manu/ST8000DM004-2U91/tmp/afd.h5',
                        help='输入的HDF5文件路径 (例如: afd.h5)')
    parser.add_argument('--train_h5', type=str, default='/home/manu/tmp/afd_pm_hdf5/train_data.h5',
                        help='输出的训练集HDF5文件路径')
    parser.add_argument('--test_h5', type=str, default='/home/manu/tmp/afd_pm_hdf5/test_data.h5',
                        help='输出的测试集HDF5文件路径')
    return parser.parse_args()


def process_h5_file(args):
    """
    读取HDF5文件，根据规则将数据分割为训练集和测试集，
    并保存到新的HDF5文件中。
    """
    # 检查输入文件是否存在
    if not os.path.exists(args.input_h5):
        logging.error(f"输入文件未找到: {args.input_h5}")
        return

    # 如果输出文件已存在，则删除，避免重复写入
    for outfile in [args.train_h5, args.test_h5]:
        if os.path.exists(outfile):
            logging.warning(f"发现已存在的文件，正在删除: {outfile}")
            os.remove(outfile)

    train_idx = 0
    test_idx = 0

    # 使用 'a' 模式（追加），如果文件不存在则会创建
    with h5py.File(args.input_h5, 'r') as h5_in, \
            h5py.File(args.train_h5, 'a') as h5_train, \
            h5py.File(args.test_h5, 'a') as h5_test:

        sample_keys = list(h5_in.keys())
        logging.info(f"在 {args.input_h5} 文件中找到 {len(sample_keys)} 个样本。")

        for i, group_name in enumerate(sample_keys):
            logging.info(f"正在处理样本 {i + 1}/{len(sample_keys)}: {group_name}")
            grp_in = h5_in[group_name]

            # 从输入组加载数据
            signal = grp_in['signal'][:]
            label_seq = grp_in['label_seq'][:]
            is_positive_sample = np.any(label_seq > 0)

            if is_positive_sample:
                # --- 这是正例样本，需要进行分割 ---

                # 1. 找到正例标签区间的起止位置
                arc_indices = np.where(label_seq > 0)[0]
                if not arc_indices.any():  # 安全检查
                    logging.warning(f"样本 {group_name} 标记为正例但未找到正例标签序列，跳过。")
                    continue
                arc_start_idx = arc_indices[0]
                arc_end_idx = arc_indices[-1]

                # 2. 定义测试数据片段（拉弧区间 + 拉弧前的正常区间）
                clip_end_idx = arc_end_idx + 1
                clip_start_idx = max(0, arc_start_idx - TEST_LENGTH)

                # 3. 如果测试片段有效，则保存到 test_data.h5
                if clip_end_idx > clip_start_idx:
                    test_signal = signal[clip_start_idx:clip_end_idx]
                    test_label_seq = label_seq[clip_start_idx:clip_end_idx]

                    grp_test = h5_test.create_group(f'sample_{test_idx}')
                    grp_test.create_dataset('signal', data=test_signal)
                    grp_test.create_dataset('label_seq', data=test_label_seq)
                    # 测试数据的摘要标签为1，因为它包含故障
                    grp_test.create_dataset('label', data=1)
                    logging.info(f"  -> 已创建测试样本 {test_idx}，长度为 {len(test_signal)}")
                    test_idx += 1

                # 4. 创建修改后的训练数据并保存到 train_data.h5
                train_label_seq = label_seq.copy()
                # 将被划分为测试数据的区间的标签设置为 2
                train_label_seq[clip_start_idx:clip_end_idx] = 2

                grp_train = h5_train.create_group(f'sample_{train_idx}')
                # 保存完整的原始信号
                grp_train.create_dataset('signal', data=signal)
                # 保存修改后的标签序列
                grp_train.create_dataset('label_seq', data=train_label_seq)
                # 整个序列的摘要标签仍然是1
                grp_train.create_dataset('label', data=1)
                logging.info(f"  -> 已创建训练样本 {train_idx} (正例, 已修改)")
                train_idx += 1

            else:
                # --- 这是负例样本，直接复制到 train_data.h5 ---
                grp_train = h5_train.create_group(f'sample_{train_idx}')
                grp_train.create_dataset('signal', data=signal)
                grp_train.create_dataset('label_seq', data=label_seq)
                grp_train.create_dataset('label', data=0)
                logging.info(f"  -> 已创建训练样本 {train_idx} (负例)")
                train_idx += 1

    logging.info("处理完成。")
    logging.info(f"{args.train_h5} 中总样本数: {train_idx}")
    logging.info(f"{args.test_h5} 中总样本数: {test_idx}")


def main():
    """主执行函数"""
    set_logging()
    args = parse_args()
    process_h5_file(args)


if __name__ == '__main__':
    main()
