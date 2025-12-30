# FILE: phase2_evaluator.py

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, \
    classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 路径修正 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 项目模块导入 ---
from utils.macros import MIN_VAL_TH
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset
from cores.classifier import MemoryHead, ArcMarginProduct

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 全量评估与参数搜索脚本")

    # 路径参数
    parser.add_argument('--path_ckpt_base', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_base.pt")
    parser.add_argument('--path_ckpt_head', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_head.pt")
    parser.add_argument('--path_ckpt_clf', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_clf.pt")
    parser.add_argument('--data_path', type=str,
                        default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3")

    # 其他配置
    parser.add_argument('--ae_model_type', type=str, default='ae', choices=['ae', 'unet'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()


class Phase2Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        logging.info(">>> 正在加载模型...")
        # 1. Base AE
        model_map = {'ae': NetAFDAE, 'unet': NetAFDAE_UNet}
        self.base_model = model_map[self.args.ae_model_type]().to(self.device).eval()
        state_base = torch.load(self.args.path_ckpt_base, map_location=self.device)
        self.base_model.load_state_dict({k.replace('module.', ''): v for k, v in state_base.items()})

        # Get latent dim
        with torch.no_grad():
            _, dummy_z, _ = self.base_model(torch.randn(1, 1, 448).to(self.device))
            latent_dim = dummy_z.shape[1]

        # 2. Memory Head
        self.mem_head = MemoryHead(latent_dim, mem_dim=1024, hidden_dim=latent_dim).to(self.device).eval()
        state_head = torch.load(self.args.path_ckpt_head, map_location=self.device)
        self.mem_head.load_state_dict({k.replace('module.', ''): v for k, v in state_head.items()})

        # 3. Aux Classifier
        self.aux_clf = ArcMarginProduct(latent_dim, 2, s=30.0, m=0.50).to(self.device).eval()
        if os.path.exists(self.args.path_ckpt_clf):
            state_clf = torch.load(self.args.path_ckpt_clf, map_location=self.device)
            self.aux_clf.load_state_dict({k.replace('module.', ''): v for k, v in state_clf.items()})
        else:
            logging.warning("未使用 AuxClassifier 权重，使用随机初始化。")

    def get_all_metrics(self):
        """
        遍历整个数据集，收集所有样本的三个指标：
        1. Recon Error
        2. Cls Probability
        3. Min Distance to Memory
        """
        logging.info(f">>> 开始全量推理: {self.args.data_path}")
        gen = FeaturesGeneratorCNN()

        # 使用 PeakAlignedDataset 确保覆盖所有潜在的故障点
        # only_normal=False 确保读取所有数据
        dataset = HDF5PeakAlignedDataset(
            hdf5_file_path=self.args.data_path,
            transform=gen.transform_sample_ae,
            seq_len=gen.seq_len,
            min_delta=MIN_VAL_TH,
            only_normal=False
        )

        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            shuffle=False, num_workers=self.args.num_workers)

        labels_list = []
        recon_list = []
        prob_list = []
        dist_list = []

        # 预先获取 Memory Bank 到 GPU，避免循环中重复传输
        mem_bank = self.mem_head.memory.detach()  # [M, D]

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Inference"):
                inputs = inputs.to(self.device)

                # 1. Base AE Forward
                recons, latents, _ = self.base_model(inputs)

                # 2. Memory Head Forward
                z = self.mem_head(latents)

                # 3. ArcFace Classifier Forward
                logits = self.aux_clf(z, label=None)
                probs = F.softmax(logits, dim=1)[:, 1]  # 取异常类的概率

                # 4. Metrics Calculation
                # (a) Recon Error (MSE)
                errs = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

                # (b) Min Distance to Memory Slots
                # z: [B, D], mem_bank: [M, D] -> dists: [B, M]
                dists_mat = torch.cdist(z, mem_bank, p=2)
                min_dists, _ = torch.min(dists_mat, dim=1)

                # Store
                labels_list.append(labels.numpy())
                recon_list.append(errs.cpu().numpy())
                prob_list.append(probs.cpu().numpy())
                dist_list.append(min_dists.cpu().numpy())

        return (
            np.concatenate(labels_list),
            np.concatenate(recon_list),
            np.concatenate(prob_list),
            np.concatenate(dist_list)
        )

    def optimize_and_evaluate(self):
        # 1. 获取原始数据
        y_true, raw_recon, raw_prob, raw_dist = self.get_all_metrics()

        logging.info(f"Total samples: {len(y_true)}")
        logging.info(f"Positives: {np.sum(y_true == 1)}, Negatives: {np.sum(y_true == 0)}")

        # 2. 数据归一化 (Min-Max Normalization)
        # 这一步非常关键，因为三个指标的量级完全不同
        # recon ~ 0.001, prob ~ 0.0-1.0, dist ~ 10.0
        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

        norm_recon = normalize(raw_recon)
        norm_prob = raw_prob  # Prob 本身就是 0-1，但分布可能偏斜，保持原样或归一化皆可，这里保持原样
        norm_dist = normalize(raw_dist)

        # 3. 定义优化目标函数
        # x = [w0, w1, w2]
        def objective_func(x):
            w0, w1, w2 = x
            # 计算加权分数
            final_scores = w0 * norm_recon + w1 * norm_prob + w2 * norm_dist

            # 自动寻找该组合下的最佳 F1 Score
            # precision_recall_curve 返回的 thresholds 是递增的
            precisions, recalls, thresholds = precision_recall_curve(y_true, final_scores)

            # 计算 F1 (防止除以0)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

            # 我们需要最大化 F1，所以返回负的最大 F1
            best_f1 = np.max(f1_scores)
            return -best_f1

        # 4. 运行差分进化算法搜索最佳权重
        logging.info(">>> 开始搜索最佳权重组合 (w0: Recon, w1: Cls, w2: Dist)...")
        bounds = [(0, 1), (0, 1), (0, 1)]  # 权重范围 0-1
        result = differential_evolution(objective_func, bounds, seed=42, maxiter=20, popsize=15)

        best_weights = result.x
        best_f1 = -result.fun

        # 归一化权重以便查看相对重要性
        total_w = np.sum(best_weights)
        norm_weights = best_weights / total_w

        logging.info(f"搜索完成! Best F1: {best_f1:.5f}")
        logging.info(
            f"Raw Weights -> w0(Recon): {best_weights[0]:.4f}, w1(Cls): {best_weights[1]:.4f}, w2(Dist): {best_weights[2]:.4f}")
        logging.info(f"Norm Weights -> w0: {norm_weights[0]:.4f}, w1: {norm_weights[1]:.4f}, w2: {norm_weights[2]:.4f}")

        # 5. 使用最佳权重进行最终评估
        w0, w1, w2 = best_weights
        final_scores = w0 * norm_recon + w1 * norm_prob + w2 * norm_dist

        # 再次寻找最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(y_true, final_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        logging.info(f"Best Threshold: {best_threshold:.6f}")

        # 生成最终预测
        y_pred = (final_scores >= best_threshold).astype(int)

        # 6. 构建返回结果
        result_dict = {
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'report': classification_report(y_true, y_pred, output_dict=True),
            'params': {
                'w0_recon': float(best_weights[0]),
                'w1_cls': float(best_weights[1]),
                'w2_dist': float(best_weights[2]),
                'threshold': float(best_threshold)
            }
        }

        return result_dict


def main():
    args = parse_args()
    evaluator = Phase2Evaluator(args)
    results = evaluator.optimize_and_evaluate()

    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(json.dumps(results, indent=4))
    print("=" * 50)


if __name__ == '__main__':
    main()
