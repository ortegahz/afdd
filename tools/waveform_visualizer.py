# FILE: waveform_explorer_aligned.py

import argparse
import logging
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button
from scipy.signal import find_peaks

# -- 路径修正 --
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.macros import MIN_VAL_TH
from cores.nets import NetAFDAE, NetAFDAE_UNet, NetAFDAE_Mem, NetAFDAE_UNet_Mem, NetAFDAE_Mem_Flow

# 硬编码序列长度
SEQ_LEN = 448


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="交互式长序列波形探索器 (仅对齐样本)")
    parser.add_argument('--model_path', type=str, default="/home/manu/mnt/8gpu_3090/afdd_models_mp/ae_best.pt")
    parser.add_argument('--data_path', type=str, default="/media/manu/ST8000DM004-2U91/tmp/afd.h5.v3")
    parser.add_argument('--ae_model_type', type=str, default='ae',
                        choices=['ae', 'unet', 'mem-ae', 'unet-mem', 'mem-flow-ae'])
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--error_threshold', type=float, default=0.001)
    return parser.parse_args()


class FeaturesGenerator:
    """仅保留推理需要的转换逻辑"""

    @staticmethod
    def transform_sample_ae(x_sample):
        """Standard AE normalization and padding"""
        x_tensor = torch.tensor(x_sample, dtype=torch.float32)
        min_val = torch.min(x_tensor)
        max_val = torch.max(x_tensor)

        if (max_val - min_val) > 0:
            x_signal = 2 * (x_tensor - min_val) / (max_val - min_val) - 1
        else:
            x_signal = torch.zeros_like(x_tensor)

        x_signal = x_signal.unsqueeze(0)  # (1, Len)

        # Padding
        curr_len = x_signal.shape[-1]
        pad_req = (32 - (curr_len % 32)) % 32
        x_signal = F.pad(x_signal, (0, pad_req), "constant", 0)

        return x_signal.unsqueeze(0)  # (1, 1, Len)


class SequenceLoader:
    def __init__(self, h5_path, model, device):
        self.h5_path = h5_path
        self.model = model
        self.device = device
        self.file = h5py.File(h5_path, 'r')
        self.keys = list(self.file.keys())
        self.current_idx = 0

        # 缓存当前长序列的推理结果
        self.current_valid_starts = []  # 存储每个合法样本的起始索引
        self.current_valid_errors = []  # 存储对应的MSE

    def get_dataset_info(self):
        return len(self.keys)

    def load_sequence(self, idx):
        """
        加载序列，找到所有峰值对齐点，并仅对这些点进行推理
        """
        key = self.keys[idx]
        group = self.file[key]
        signal = group['signal'][:]
        label_seq = group['label_seq'][:]

        # 1. 寻找合法对齐点 (逻辑同 HDF5PeakAlignedDataset)
        # peaks 是峰值的索引
        peaks, _ = find_peaks(signal, distance=SEQ_LEN // 2, prominence=MIN_VAL_TH * 2)

        valid_starts = []
        batch_inputs = []

        for p in peaks:
            # 样本区间是 [p + 1 - SEQ_LEN, p + 1]
            end_idx = p + 1
            start_idx = end_idx - SEQ_LEN

            if start_idx >= 0:
                valid_starts.append(start_idx)
                # 准备数据进行推理
                sample = signal[start_idx:end_idx]
                tensor = FeaturesGenerator.transform_sample_ae(sample)
                batch_inputs.append(tensor)

        valid_starts = np.array(valid_starts)
        errors = np.zeros(len(valid_starts))

        # 2. 批量推理计算误差
        if batch_inputs:
            batch_tensor = torch.cat(batch_inputs, dim=0).to(self.device)
            # 分批推理防止显存溢出
            batch_size = 256
            all_mses = []

            with torch.no_grad():
                for i in range(0, len(batch_tensor), batch_size):
                    b = batch_tensor[i: i + batch_size]
                    recons = self.model(b)[0]
                    mse = torch.mean((b - recons) ** 2, dim=(1, 2)).cpu().numpy()
                    all_mses.append(mse)

            if all_mses:
                errors = np.concatenate(all_mses)

        # 缓存结果供交互使用
        self.current_valid_starts = valid_starts
        self.current_valid_errors = errors

        return key, signal, label_seq, valid_starts, errors

    def get_nearest_sample_detail(self, signal, mouse_x):
        """
        找到距离 mouse_x 最近的合法起始点，并返回详情
        """
        if len(self.current_valid_starts) == 0:
            return None, None, None, 0.0, 0

        # Snap to nearest valid anchor
        idx = (np.abs(self.current_valid_starts - mouse_x)).argmin()
        start_idx = self.current_valid_starts[idx]
        mse = self.current_valid_errors[idx]

        # 实时生成一下波形图用于显示 (虽然之前推理过，但没存recons本身以省内存)
        seg = signal[start_idx: start_idx + SEQ_LEN]
        tensor = FeaturesGenerator.transform_sample_ae(seg).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            recon = outputs[0]

        return seg, tensor.cpu().numpy().flatten(), recon.cpu().numpy().flatten(), mse, start_idx


class WaveformExplorer:
    def __init__(self, loader, args):
        self.loader = loader
        self.args = args

        self.curr_signal = None
        self.curr_labels = None
        self.valid_starts = None
        self.valid_errors = None
        self.curr_key = ""

        # Setup Figure
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("AFDD Aligned Explorer (Snapping Enabled)", fontsize=16)

        # Layouts
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.2, 2, 0.2])
        self.ax_full = self.fig.add_subplot(gs[0, :])
        self.ax_detail = self.fig.add_subplot(gs[1, 0])
        self.ax_diff = self.fig.add_subplot(gs[1, 1], sharex=self.ax_detail)

        # Controls
        self.ax_prev = self.fig.add_subplot(gs[2, 0])
        self.ax_next = self.fig.add_subplot(gs[2, 1])
        # Hide axes frames for buttons container
        self.ax_prev.axis('off')
        self.ax_next.axis('off')

        # Create Buttons (manually positioning axis)
        ax_b1 = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_b2 = plt.axes([0.6, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_b1, '<< Prev Sequence')
        self.btn_next = Button(ax_b2, 'Next Sequence >>')
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)

        # State variables for plots
        self.vline = self.ax_full.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        self.highlight_box = None
        self.last_snap_idx = -1  # 防止重复绘图

        # Events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Init
        self.load_current_index()

    def load_current_index(self):
        logging.info(f"Loading sequence {self.loader.current_idx + 1}/{self.loader.get_dataset_info()}...")
        self.curr_key, self.curr_signal, self.curr_labels, self.valid_starts, self.valid_errors = \
            self.loader.load_sequence(self.loader.current_idx)

        self.plot_full_sequence()
        # Default show first valid sample or 0
        init_pos = self.valid_starts[0] if len(self.valid_starts) > 0 else 0
        self.update_detail_plot(init_pos)

    def plot_full_sequence(self):
        self.ax_full.clear()
        seq_len = len(self.curr_signal)
        x = np.arange(seq_len)

        # 1. Base Signal
        self.ax_full.plot(x, self.curr_signal, color='gray', alpha=0.4, linewidth=0.8, label='Signal')

        # 2. Fault Regions (Background)
        fault_mask = self.curr_labels > 0
        if np.any(fault_mask):
            self.ax_full.fill_between(x, np.min(self.curr_signal), np.max(self.curr_signal),
                                      where=fault_mask, color='mistyrose', alpha=0.5, label='Fault Zone')

        # 3. Valid Anchors (Green Markers) - Show where we can snap
        # We plot markers at the END of the window (the peak) because that's visually cleaner
        if len(self.valid_starts) > 0:
            peak_locs = self.valid_starts + SEQ_LEN
            self.ax_full.plot(peak_locs, self.curr_signal[peak_locs], 'v', color='limegreen',
                              markersize=4, label='Aligned Anchor (Snap Point)')

            # 4. Highlight High Error Anchors (Gold Dots)
            high_err_indices = np.where(self.valid_errors > self.args.error_threshold)[0]
            if len(high_err_indices) > 0:
                high_err_starts = self.valid_starts[high_err_indices]
                high_err_peaks = high_err_starts + SEQ_LEN
                # 同样标记在峰值位置
                self.ax_full.scatter(high_err_peaks, self.curr_signal[high_err_peaks],
                                     c='gold', s=30, zorder=5, label='High Error Anchor')

        self.ax_full.set_title(f"Seq: {self.curr_key} | Green=Allowed, Gold=Hard/Fault | Mouse snaps to Green",
                               fontsize=10)
        self.ax_full.set_xlim(0, seq_len)
        self.ax_full.legend(loc='upper right', ncol=4, fontsize='small')

        # Reset indicators
        self.vline = self.ax_full.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        self.highlight_box = None

    def update_detail_plot(self, mouse_x):
        # Find nearest valid sample
        orig, input_norm, recon, mse, start_idx = \
            self.loader.get_nearest_sample_detail(self.curr_signal, mouse_x)

        if orig is None: return

        # Avoid redrawing if we snapped to the same index
        if start_idx == self.last_snap_idx:
            return
        self.last_snap_idx = start_idx

        # Check label for this specific window
        is_fault = np.any(self.curr_labels[start_idx: start_idx + SEQ_LEN] > 0)
        is_hard = mse > self.args.error_threshold

        # --- Detail Plot ---
        self.ax_detail.clear()
        self.ax_detail.plot(input_norm, color='black', label='Input (Norm)', linewidth=1.5)
        self.ax_detail.plot(recon, color='cyan', linestyle='--', label='Recon', linewidth=1.5)

        title_color = 'black'
        status_text = "NORMAL (Easy)"
        bg_color = 'white'

        if is_fault:
            title_color = 'white'
            status_text = "!!! FAULT !!!"
            bg_color = 'firebrick'
        elif is_hard:
            title_color = 'black'
            status_text = "HARD NORMAL (High Error)"
            bg_color = 'gold'

        self.ax_detail.set_title(
            f"Aligned Window [{start_idx}:{start_idx + SEQ_LEN}] | MSE: {mse:.6f}\nStatus: {status_text}",
            color=title_color, backgroundcolor=bg_color, fontweight='bold')
        self.ax_detail.legend(loc='upper right')
        self.ax_detail.grid(True, linestyle=':', alpha=0.6)

        # --- Diff Plot ---
        self.ax_diff.clear()
        diff = np.abs(input_norm - recon)
        self.ax_diff.plot(diff, color='magenta', alpha=0.9)
        self.ax_diff.fill_between(np.arange(SEQ_LEN), diff, color='magenta', alpha=0.2)
        self.ax_diff.set_title("Residue")
        self.ax_diff.grid(True, linestyle=':', alpha=0.6)
        self.ax_diff.set_ylim(0, max(np.max(diff) * 1.1, 0.1))

        # --- Update Visual Indicators on Top Plot ---
        self.vline.set_xdata([start_idx + SEQ_LEN])  # Line at peak

        if self.highlight_box: self.highlight_box.remove()
        self.highlight_box = self.ax_full.axvspan(start_idx, start_idx + SEQ_LEN, color='blue', alpha=0.15)

        self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if not event.inaxes: return
        if event.inaxes == self.ax_full:
            # Pass raw mouse position, loader will snap it
            self.update_detail_plot(event.xdata)

    def on_prev(self, event):
        if self.loader.current_idx > 0:
            self.loader.current_idx -= 1
            self.last_snap_idx = -1  # Reset snap state
            self.load_current_index()

    def on_next(self, event):
        if self.loader.current_idx < self.loader.get_dataset_info() - 1:
            self.loader.current_idx += 1
            self.last_snap_idx = -1
            self.load_current_index()


def main():
    setup_logging()
    args = parse_args()

    # 1. Model
    device = torch.device(args.device)
    logging.info(f"Loading Model: {args.ae_model_type}")

    model_map = {
        'ae': NetAFDAE, 'unet': NetAFDAE_UNet,
        'mem-ae': NetAFDAE_Mem, 'unet-mem': NetAFDAE_UNet_Mem,
        'mem-flow-ae': NetAFDAE_Mem_Flow
    }
    model = model_map[args.ae_model_type]()

    try:
        sd = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return

    model.to(device)
    model.eval()

    # 2. Loader
    if not os.path.exists(args.data_path):
        logging.error("Data path not found")
        return

    loader = SequenceLoader(args.data_path, model, device)

    # 3. UI
    logging.info("Starting Explorer (Aligned)...")
    logging.info("Move mouse over top plot. It will SNAP to the nearest valid sample.")
    explorer = WaveformExplorer(loader, args)
    plt.show()


if __name__ == '__main__':
    main()
