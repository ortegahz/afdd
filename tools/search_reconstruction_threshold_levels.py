# FILE: search_reconstruction_threshold_levels.py

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 路径修正 (确保能导入项目模块) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 项目模块导入 ---
from utils.macros import MIN_VAL_TH
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset, HDF5ArcFaultDataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="搜索不同 Recall (100/99/98/97%) 下的重构阈值")

    # 路径参数
    parser.add_argument('--path_ckpt_base', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/ae_best.pt",
                        help="Base AE 模型权重路径")
    parser.add_argument('--data_path', type=str,
                        default="/home/manu/mnt/8gpu_3090/afd_pm_hdf5/train_data.h5",
                        help="HDF5 数据集路径")

    # 其他配置
    parser.add_argument('--ae_model_type', type=str, default='ae', choices=['ae', 'unet'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info(f"使用设备: {device}")
    logging.info(f"数据路径: {args.data_path}")

    # 1. 加载 Base AE 模型
    logging.info(">>> 正在加载模型...")
    model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
    model = model_map[args.ae_model_type]().to(device).eval()

    if not os.path.exists(args.path_ckpt_base):
        logging.error(f"模型权重文件不存在: {args.path_ckpt_base}")
        return

    state_base = torch.load(args.path_ckpt_base, map_location=device)
    # 处理可能的 DataParallel 'module.' 前缀
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_base.items()})

    # 2. 准备数据加载器
    logging.info(">>> 正在准备数据...")
    gen = FeaturesGeneratorCNN()
    # dataset = HDF5PeakAlignedDataset(
    #     hdf5_file_path=args.data_path,
    #     transform=gen.transform_sample_ae,
    #     seq_len=gen.seq_len,
    #     min_delta=MIN_VAL_TH
    # )
    dataset = HDF5ArcFaultDataset(args.data_path)

    # 使用较大的 batch_size 加速推理，不需要 shuffle
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. 推理并收集误差
    logging.info(">>> 开始推理计算重构误差...")
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Scanning Dataset"):
            inputs = inputs.to(device)

            # Forward pass
            recons, _, _ = model(inputs)

            # 计算 MSE Loss: mean over (channels, length) -> [Batch_size]
            # 假设 input shape 是 [B, 1, 448]
            errs = torch.mean((inputs - recons) ** 2, dim=(1, 2))

            all_errors.append(errs.cpu().numpy())
            all_labels.append(labels.numpy())

    # 4. 数据整合
    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels).reshape(-1)

    # 提取异常样本 (Label == 1) 和 正常样本 (Label == 0)
    abnormal_mask = (labels == 1)
    normal_mask = (labels == 0)

    abnormal_errors = errors[abnormal_mask]
    normal_errors = errors[normal_mask]

    num_abnormal = len(abnormal_errors)
    num_normal = len(normal_errors)

    logging.info("-" * 50)
    logging.info(f"正常样本总数: {num_normal}")
    logging.info(f"异常样本总数: {num_abnormal}")
    logging.info("-" * 50)

    if num_abnormal == 0:
        logging.warning("数据集中未找到异常样本 (Label=1)，无法计算阈值。")
        return

    # 5. 计算不同 Recall 下的阈值
    # Recall X% 意味着我们要保留 X% 的异常样本被判定为异常（Error >= Threshold）
    # 这等价于寻找异常样本误差分布的 (100 - X) 分位数

    target_recalls = [100, 99, 98, 97]

    print("\n====== 阈值搜索结果 (Threshold Search Results) ======")
    print(f"{'Target Recall':<15} | {'Threshold':<15} | {'False Positives':<15} | {'FPR (%)':<10}")
    print("-" * 70)

    for recall in target_recalls:
        # 计算分位数 q
        # 100% Recall -> 0th percentile (Min value)
        # 99% Recall -> 1st percentile
        q = 100 - recall

        # 获取阈值
        threshold = np.percentile(abnormal_errors, q)

        # 统计在该阈值下的误报情况 (正常样本中 Error >= Threshold 的数量)
        false_positives = np.sum(normal_errors >= threshold)
        fpr = (false_positives / num_normal) * 100 if num_normal > 0 else 0.0

        print(f"{recall}%             | {threshold:.8f}      | {false_positives:<15} | {fpr:.4f}%")

    print("-" * 70)
    print("注: Threshold = 异常样本误差的 (100 - Recall) 分位数")
    print("    False Positives = 正常样本中误差大于该阈值的数量")
    print("    FPR = False Positives / Total Normal Samples")


if __name__ == "__main__":
    main()
