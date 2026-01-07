# FILE: phase2_eval_two_stage_fixed_v2.py

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.macros import RECONS_TH

# --- 路径修正 (确保能导入项目模块) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 项目模块导入 ---
from utils.macros import MIN_VAL_TH
from cores.nets import NetAFDAE, NetAFDAE_UNet
from cores.features_generator import FeaturesGeneratorCNN, HDF5PeakAlignedDataset, HDF5ArcFaultDataset
from cores.classifier import MemoryHead, ArcMarginProduct

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 两阶段评估 (固定重构阈值)")

    # 路径参数
    parser.add_argument('--path_ckpt_base', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_base.pt")
    parser.add_argument('--path_ckpt_head', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_head.pt")
    parser.add_argument('--path_ckpt_clf', type=str,
                        default="/home/manu/mnt/8gpu_3090/afdd_models_mp/phase2_best_clf.pt")
    parser.add_argument('--data_path', type=str,
                        default="/home/manu/mnt/8gpu_3090/afd_pm_hdf5/test_data.h5")

    # 模型与计算配置
    parser.add_argument('--ae_model_type', type=str, default='ae', choices=['ae', 'unet'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=512)

    # 策略参数
    parser.add_argument('--fixed_recon_th', type=float, default=RECONS_TH,
                        help="第一阶段固定的重构误差阈值 (Raw MSE)")
    parser.add_argument('--weight_step', type=float, default=0.05, help="第二阶段权重搜索步长")

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
        state_base = torch.load(self.args.path_ckpt_base, map_location=self.device, weights_only=False)
        self.base_model.load_state_dict({k.replace('module.', ''): v for k, v in state_base.items()})

        # Get latent dim
        with torch.no_grad():
            _, dummy_z, _ = self.base_model(torch.randn(1, 1, 448).to(self.device))
            latent_dim = dummy_z.shape[1]

        # 2. Memory Head
        self.mem_head = MemoryHead(latent_dim, mem_dim=1024, hidden_dim=latent_dim).to(self.device).eval()
        state_head = torch.load(self.args.path_ckpt_head, map_location=self.device, weights_only=False)
        self.mem_head.load_state_dict({k.replace('module.', ''): v for k, v in state_head.items()})

        # 3. Aux Classifier
        self.aux_clf = ArcMarginProduct(latent_dim, 2, s=30.0, m=0.50).to(self.device).eval()
        if os.path.exists(self.args.path_ckpt_clf):
            state_clf = torch.load(self.args.path_ckpt_clf, map_location=self.device, weights_only=False)
            self.aux_clf.load_state_dict({k.replace('module.', ''): v for k, v in state_clf.items()})
        else:
            logging.warning("未使用 AuxClassifier 权重，使用随机初始化。")

    def extract_all_scores(self):
        """
        遍历整个数据集，提取三个维度的原始分数
        """
        logging.info(">>> 开始全量推理与特征提取...")
        gen = FeaturesGeneratorCNN()
        # dataset = HDF5PeakAlignedDataset(
        #     hdf5_file_path=self.args.data_path,
        #     transform=gen.transform_sample_ae,
        #     seq_len=gen.seq_len,
        #     min_delta=MIN_VAL_TH
        # )
        dataset = HDF5ArcFaultDataset(self.args.data_path)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        all_labels = []
        raw_recon_errs = []
        raw_mem_dists = []
        raw_cls_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Inference"):
                inputs = inputs.to(self.device)

                # 1. Base AE Forward
                recons, latents, _ = self.base_model(inputs)

                # 2. Memory Head Forward
                z = self.mem_head(latents)

                # 3. Classifier Forward
                logits = self.aux_clf(z, label=None)
                probs = F.softmax(logits, dim=1)[:, 1]  # 取异常类的概率

                # --- Metrics Calculation ---

                # A. Recon Error (MSE)
                errs = torch.mean((inputs - recons) ** 2, dim=tuple(range(1, inputs.dim())))

                # B. Memory Distance (Min Euclidean Dist)
                mem_bank = self.mem_head.memory
                dists = torch.cdist(z, mem_bank, p=2)  # [B, M]
                min_dists, _ = torch.min(dists, dim=1)

                # Collect
                all_labels.append(labels.numpy())
                raw_recon_errs.append(errs.cpu().numpy())
                raw_mem_dists.append(min_dists.cpu().numpy())
                raw_cls_probs.append(probs.cpu().numpy())

        # Concatenate
        self.labels = np.concatenate(all_labels).flatten()
        self.recon_errs = np.concatenate(raw_recon_errs)
        self.mem_dists = np.concatenate(raw_mem_dists)
        self.cls_probs = np.concatenate(raw_cls_probs)

        logging.info(f"推理完成。总样本数: {len(self.labels)}")
        logging.info(f"正例数: {np.sum(self.labels == 1)}, 负例数: {np.sum(self.labels == 0)}")

    def search_optimal_params(self):
        """
        两阶段参数搜索：
        1. 固定 Recon Threshold (args.fixed_recon_th)
        2. 对于 > fixed_recon_th 的样本，搜索最佳权重 (w_dist, w_cls) 和 组合阈值 (th_combined)
        """
        logging.info(f">>> 开始参数搜索 (Fixed Recon Threshold: {self.args.fixed_recon_th})...")

        # 1. 第一阶段筛选 (使用原始 MSE 值)
        # Mask: 找出第一层判定为“可疑”的样本 (Recon > Fixed Threshold)
        mask_suspicious = self.recon_errs > self.args.fixed_recon_th

        suspicious_count = np.sum(mask_suspicious)
        logging.info(f"第一阶段筛选后，进入第二阶段的可疑样本数: {suspicious_count} / {len(self.labels)}")

        if suspicious_count == 0:
            logging.warning("没有样本超过重构阈值，所有样本被预测为 Normal。")
            return {'w_dist': 0.5, 'w_cls': 0.5, 'th_combined': 0.5}

        # 2. 归一化第二阶段指标 (Normalization)
        scaler = MinMaxScaler()
        norm_dist = scaler.fit_transform(self.mem_dists.reshape(-1, 1)).flatten()
        norm_cls = self.cls_probs

        # 保存归一化后的数据供 evaluate_final 使用
        self.norm_dist = norm_dist
        self.norm_cls = norm_cls

        # 3. 定义搜索空间：权重组合 (w_dist, w_cls)
        weights = np.arange(0, 1.0 + 1e-5, self.args.weight_step)

        best_f1 = -1.0
        best_params = {}

        # 4. 遍历权重搜索
        for w_d in tqdm(weights, desc="Searching Weights"):
            w_c = 1.0 - w_d

            # 计算第二层组合分数: w_dist * Dist + w_cls * Prob
            combined_scores_raw = (w_d * norm_dist) + (w_c * norm_cls)

            # --- 构造用于寻找最佳 th_combined 的合成全量分数 ---
            # 逻辑：
            # 如果 Recon <= fixed_th: 预测必须为0。
            # 为了避免 sklearn 报错，我们使用 -1.0 代替 -inf。
            # 因为 norm_dist 和 norm_cls 都在 [0, 1]，所以 combined_scores_raw >= 0。
            # 设置为 -1.0 保证了这些样本在任何合理的正数阈值下都会被判定为 0。

            final_scores_for_search = np.full_like(combined_scores_raw, -1.0)
            final_scores_for_search[mask_suspicious] = combined_scores_raw[mask_suspicious]

            # 使用 precision_recall_curve 快速找到当前权重下的最佳 F1 和 th_combined
            precisions, recalls, thresholds = precision_recall_curve(self.labels, final_scores_for_search)

            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores)

            idx = np.argmax(f1_scores)
            current_max_f1 = f1_scores[idx]

            if current_max_f1 > best_f1:
                best_f1 = current_max_f1
                # 获取对应的 th_combined
                best_th_c = thresholds[idx] if idx < len(thresholds) else thresholds[-1]

                best_params = {
                    'w_dist': w_d,  # 距离权重
                    'w_cls': w_c,  # 概率权重
                    'th_combined': best_th_c  # 第二层阈值
                }

        logging.info("-" * 50)
        logging.info(f"搜索结束。最佳 F1 Score: {best_f1:.6f}")
        logging.info(f"最佳参数: {best_params}")

        return best_params

    def evaluate_final(self, params):
        """
        使用最佳参数计算最终指标
        """
        fixed_th = self.args.fixed_recon_th
        w_d = params['w_dist']
        w_c = params['w_cls']
        th_c = params['th_combined']

        # 1. 第一层筛选 (Raw MSE)
        mask_suspicious = self.recon_errs > fixed_th

        # 初始化预测全为 0 (Normal)
        preds = np.zeros_like(self.labels, dtype=int)

        # 2. 第二层判定
        if np.sum(mask_suspicious) > 0:
            # 计算组合分 (Normalized features)
            scores_stage2 = (w_d * self.norm_dist) + (w_c * self.norm_cls)

            # 逻辑：是可疑样本 AND 组合分 > 第二层阈值 -> Abnormal (1)
            is_abnormal = (scores_stage2 > th_c) & mask_suspicious
            preds[is_abnormal] = 1

        precision = precision_score(self.labels, preds)
        recall = recall_score(self.labels, preds)
        f1 = f1_score(self.labels, preds)
        accuracy = accuracy_score(self.labels, preds)
        report = classification_report(self.labels, preds, target_names=['Normal', 'Abnormal'])

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'report': report,
            'best_params': params
        }


def main():
    args = parse_args()

    # 1. 初始化与推理
    evaluator = Phase2Evaluator(args)
    evaluator.extract_all_scores()

    # 2. 搜索最优参数 (固定第一层，搜索第二层)
    best_params = evaluator.search_optimal_params()

    # 3. 计算最终指标
    results = evaluator.evaluate_final(best_params)

    # 4. 打印报告
    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT (Two-Stage Fixed Threshold)")
    print("=" * 60)
    print("Strategy:")
    print(f"  1. If Recon_Error <= {args.fixed_recon_th}: Predict Normal")
    print("  2. If Recon_Error >  Threshold: Check (w_dist*Dist + w_cls*Prob) > th_combined")
    print("-" * 60)
    print(f"Fixed Parameters:")
    print(f"  - Stage 1 Threshold (Recon): {args.fixed_recon_th}")
    print(f"Optimal Parameters (Stage 2):")
    print(f"  - Weight (Dist):             {best_params['w_dist']:.2f}")
    print(f"  - Weight (Cls):              {best_params['w_cls']:.2f}")
    print(f"  - Threshold (Combined):      {best_params['th_combined']:.6f}")
    print("-" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(results['report'])
    print("=" * 60)


if __name__ == "__main__":
    main()
