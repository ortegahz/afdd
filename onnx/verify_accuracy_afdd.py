#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the accuracy of a float32 ONNX model and its quantized version
for an input tensor of shape (1, 1, 448).
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error)
from tqdm import tqdm

try:
    from fastdtw import fastdtw  # optional

    HAS_FDTW = True
except ImportError:
    HAS_FDTW = False

_EPS = 1e-10


# -------------------------------------------------
# 数据预处理 —— 生成 (1, 1, 448)
# -------------------------------------------------
def preprocess(img_path: str, vec_len: int = 448) -> np.ndarray:
    """
    读取图片, 转为灰度, resize 为 (vec_len, 1), 返回形状 (1, 1, 448) float32.
    """
    img = Image.open(img_path).convert('L')  # 灰度
    img = img.resize((vec_len, 1), Image.BILINEAR)  # 宽 448 × 高 1

    # (1, 448) -> (1, 1, 448)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, vec_len)  # (1, 448)
    arr = np.expand_dims(arr, 0)  # (1, 1, 448)
    return arr


def load_labels(label_file: str):
    mapping = {}
    with open(label_file, 'r') as f:
        for line in f:
            name, lab = line.strip().split()
            mapping[name] = int(lab)
    return mapping


# -------------------------------------------------
# 推理
# -------------------------------------------------
def run_session(session: ort.InferenceSession,
                input_name: str,
                img_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    preds, probs = [], []
    for img in img_batch:
        logits = session.run(None, {input_name: img})[0]  # (1, C)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        preds.append(np.argmax(prob, axis=1))
        probs.append(prob)
    return np.concatenate(preds, axis=0), np.concatenate(probs, axis=0)


# -------------------------------------------------
# Metric helpers
# -------------------------------------------------
def topk_correct(probs: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([t in topk[i] for i, t in enumerate(targets)])


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - np.max(v))
    return e / (np.sum(e) + _EPS)


def calc_metrics(fp32_probs, int8_probs, fp32_preds,
                 int8_preds, labels_provided, gt=None):
    metrics = {}

    # Difference
    diff = fp32_probs - int8_probs
    metrics['MSE'] = mean_squared_error(fp32_probs, int8_probs)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(fp32_probs, int8_probs)
    metrics['MedianAE'] = median_absolute_error(fp32_probs, int8_probs)
    metrics['MaxError'] = float(np.max(np.abs(diff)))
    metrics['MBE'] = np.mean(int8_probs - fp32_probs)  # Bias

    # Cosine similarity
    flat1, flat2 = fp32_probs.flatten(), int8_probs.flatten()
    metrics['CosineSim'] = 1 - cosine(flat1, flat2)

    # Robust losses
    delta = 1.0
    huber = np.where(np.abs(diff) <= delta,
                     0.5 * diff ** 2,
                     delta * (np.abs(diff) - 0.5 * delta))
    metrics['Huber'] = np.mean(huber)
    metrics['LogCosh'] = np.mean(np.log(np.cosh(diff)))

    # JS divergence
    js_list = []
    for p, q in zip(fp32_probs, int8_probs):
        p = _softmax(p)
        q = _softmax(q)
        js = 0.5 * (entropy(p, q) + entropy(q, p))
        js_list.append(js)
    metrics['MeanJS'] = float(np.mean(js_list))

    # DTW
    if HAS_FDTW:
        dtw_dist, _ = fastdtw(flat1, flat2)
        metrics['DTW'] = dtw_dist

    # Top-k accuracy
    if labels_provided:
        metrics['Top1_FP32'] = float(np.mean(fp32_preds == gt))
        metrics['Top1_INT8'] = float(np.mean(int8_preds == gt))
        metrics['ΔTop1'] = metrics['Top1_FP32'] - metrics['Top1_INT8']

        metrics['Top5_FP32'] = topk_correct(fp32_probs, gt, 5)
        metrics['Top5_INT8'] = topk_correct(int8_probs, gt, 5)
        metrics['ΔTop5'] = metrics['Top5_FP32'] - metrics['Top5_INT8']
    return metrics


def print_metrics(metrics: dict, labels_provided: bool):
    print("\n========== Metric Summary ==========")
    if labels_provided:
        print(f"Top-1  FP32: {metrics['Top1_FP32']:.4f}  "
              f"INT8: {metrics['Top1_INT8']:.4f}  "
              f"Δ: {metrics['ΔTop1']:+.4f}")
        print(f"Top-5  FP32: {metrics['Top5_FP32']:.4f}  "
              f"INT8: {metrics['Top5_INT8']:.4f}  "
              f"Δ: {metrics['ΔTop5']:+.4f}")
    print(f"MSE:        {metrics['MSE']:.6e}")
    print(f"RMSE:       {metrics['RMSE']:.6e}")
    print(f"MAE:        {metrics['MAE']:.6e}")
    print(f"Median AE:  {metrics['MedianAE']:.6e}")
    print(f"Max Error:  {metrics['MaxError']:.6e}")
    print(f"MBE:        {metrics['MBE']:.6e}")
    print(f"Cosine Sim: {metrics['CosineSim']:.6f}")
    print(f"Huber Loss: {metrics['Huber']:.6e}")
    print(f"Log-Cosh:   {metrics['LogCosh']:.6e}")
    print(f"Mean JS Divergence: {metrics['MeanJS']:.6e}")
    if 'DTW' in metrics:
        print(f"DTW Dist.:  {metrics['DTW']:.6e}")
    else:
        print("DTW Dist.:  fastdtw 未安装，未计算")


# -------------------------------------------------
# main
# -------------------------------------------------
def main(args):
    # 1. collect images
    img_files = sorted([f for f in os.listdir(args.data_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    labels = load_labels(args.label_file) if args.label_file else None
    if labels:
        img_files = [f for f in img_files if f in labels]

    if not img_files:
        raise RuntimeError("未找到可用测试图片！")

    # 2. ORT Sessions
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess_fp32 = ort.InferenceSession(args.float_model, so, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(args.quant_model, so, providers=['CPUExecutionProvider'])
    input_name = sess_fp32.get_inputs()[0].name

    # 3. inference
    fp32_preds, int8_preds = [], []
    fp32_probs, int8_probs = [], []

    for img_name in tqdm(img_files, desc='Inference'):
        img_np = preprocess(os.path.join(args.data_dir, img_name))
        p1, prob1 = run_session(sess_fp32, input_name, [img_np])
        p2, prob2 = run_session(sess_int8, input_name, [img_np])

        fp32_preds.append(p1)
        int8_preds.append(p2)
        fp32_probs.append(prob1)
        int8_probs.append(prob2)

    fp32_preds = np.concatenate(fp32_preds, axis=0)
    int8_preds = np.concatenate(int8_preds, axis=0)
    fp32_probs = np.concatenate(fp32_probs, axis=0)
    int8_probs = np.concatenate(int8_probs, axis=0)

    # 4. metrics
    gt = np.array([labels[f] for f in img_files], dtype=np.int64) if labels else None
    metrics = calc_metrics(fp32_probs, int8_probs,
                           fp32_preds, int8_preds,
                           labels_provided=bool(labels), gt=gt)

    # 5. print
    print_metrics(metrics, labels_provided=bool(labels))


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check accuracy loss after ONNX QDQ quantization (input 1×1×448)')
    parser.add_argument('--float_model',
                        default="/home/manu/tmp/classifier_sim.onnx")
    parser.add_argument('--quant_model',
                        default="/home/manu/tmp/classifier_sim-quant.onnx")
    parser.add_argument('--data_dir',
                        default="/home/manu/tmp/test_images/")
    parser.add_argument('--label_file',
                        default=None,
                        help='optional label file (txt) for accuracy computation')
    args = parser.parse_args()
    main(args)
