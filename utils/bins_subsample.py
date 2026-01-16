#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_and_plot_all_bins.py
-------------------------------------------------
递归解析目录下全部 1 MSPS .bin 文件，
降采样(抽取)到 20 kSPS，保存新的 .bin，
并可选地绘制波形。
"""

import argparse
import gc
import os
import struct
from pathlib import Path
from typing import Optional, Tuple, Generator

import matplotlib.pyplot as plt
import numpy as np

from utils import make_dirs

# 有效位数（只影响绘图的 y 轴范围）
N_BIT_VALID = 12

# -------------------- 抽取相关参数 --------------------
FS_IN = 1_000_000  # 原采样率 1 MSPS
FS_OUT = 20_000  # 目标采样率 20 kSPS
DECIM = FS_IN // FS_OUT  # =50；要求能整除
assert FS_IN % FS_OUT == 0, "FS_IN 必须能被 FS_OUT 整除"


# -----------------------------------------------------

# =====================================================
# 读 / 写 / 抽取
# =====================================================
def parse_one_bin(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    把单个 bin 文件解析成两个 numpy 数组 (odd, even)
    odd  : 数据包中前 2 字节
    even : 数据包中后 2 字节
    """
    odd_values, even_values = [], []

    with open(file_path, "rb") as f:
        data = f.read()

    # 每 4 字节为一组
    for i in range(0, len(data) - 3, 4):
        v1 = struct.unpack_from("<H", data, i)[0]  # odd
        v2 = struct.unpack_from("<H", data, i + 2)[0]  # even
        odd_values.append(v1)
        even_values.append(v2)

    return (np.asarray(odd_values, dtype=np.uint16),
            np.asarray(even_values, dtype=np.uint16))


def decimate_two_channels(odd: np.ndarray,
                          even: np.ndarray,
                          factor: int = DECIM) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单抽取：每 factor 点保留 1 点。
    如需抗混叠滤波可替换为 scipy.signal.decimate / resample_poly。
    """
    return odd[::factor], even[::factor]


def save_two_channels_to_bin(odd: np.ndarray,
                             even: np.ndarray,
                             dst_path: Path) -> None:
    """
    把 odd / even 两通道重新交织后写入 dst_path：
        odd0 even0 odd1 even1 ...
    每个采样点为 uint16 little-endian。
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.stack((odd, even), axis=1).ravel().astype(np.uint16)
    with open(dst_path, "wb") as f:
        interleaved.tofile(f)
    print(f"  -> 新 bin 保存到 {dst_path}")


# =====================================================
# 绘图
# =====================================================
def plot_arrays(odd_arr: np.ndarray,
                even_arr: np.ndarray,
                title: str,
                save_dir: Optional[Path] = None,
                show: bool = False) -> None:
    """
    根据 odd / even 两组数据画图。
    save_dir 为 None 时只弹窗（若 show=True）；否则保存 PNG。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(even_arr, "r-")
    ax1.set_title("UAC (Even)")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("Value")
    ax1.set_ylim(0, 2 ** N_BIT_VALID)
    ax1.grid(True, alpha=0.3)

    ax2.plot(odd_arr, "b-")
    ax2.set_title("ARC (Odd)")
    ax2.set_xlabel("sample")
    ax2.set_ylabel("Value")
    ax2.set_ylim(0, 2 ** N_BIT_VALID)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        png_name = save_dir / (Path(title).stem + ".png")
        fig.savefig(png_name, dpi=150)
        print(f"  -> 已保存图像 {png_name}")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================
# 遍历目录
# =====================================================
def find_all_bin_files(root_dir: Path) -> Generator[Path, None, None]:
    """递归生成 root_dir 下全部 .bin 文件路径"""
    for path, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".bin"):
                yield Path(path) / fn


# =====================================================
# 主程序
# =====================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="递归解析、抽取并绘制目录下全部 .bin 文件")
    # >>>>>>>>>  保持你的默认 root / out  <<<<<<<<<
    parser.add_argument("--root",
                        default="/media/manu/ST8000DM004-2U91/afdd/data/data_v37/",
                        help="含有原始 1 MSPS bin 的根目录")
    parser.add_argument("--out",
                        default="/media/manu/ST8000DM004-2U91/tmp/",
                        help="抽取后 bin 输出目录")
    # 和以前一样：默认 True
    parser.add_argument("--show",
                        default=False,
                        help="是否弹窗显示波形 (默认 True)")

    args = parser.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not root_dir.exists():
        print(f"目录不存在: {root_dir}")
        return

    bin_files = list(find_all_bin_files(root_dir))
    if not bin_files:
        print("未找到任何 .bin 文件")
        return

    print(f"共找到 {len(bin_files)} 个 .bin 文件\n")

    make_dirs(out_dir, reset=True)

    for idx, bin_path in enumerate(bin_files, 1):
        # if idx < 41:
        #     continue

        print(f"[{idx:03d}/{len(bin_files)}] 处理 {bin_path}")

        # 1) 解析
        odd_arr, even_arr = parse_one_bin(bin_path)
        print(f"   原尺寸 {odd_arr.size} 点 -> ", end="")

        # 2) 抽取
        odd_ds, even_ds = decimate_two_channels(odd_arr, even_arr)
        print(f"抽取后 {odd_ds.size} 点")

        # 3) 保存新 bin（文件名加 _20k）
        rel_path = bin_path.relative_to(root_dir)
        new_name = rel_path.stem + "_20k" + rel_path.suffix
        new_path = out_dir / rel_path.parent / new_name
        save_two_channels_to_bin(odd_ds, even_ds, new_path)

        gc.collect()

        if not args.show:
            continue

        # 4) 绘图（可选）
        plot_arrays(odd_ds,
                    even_ds,
                    title=str(new_path),
                    save_dir=None,  # 如需保存 PNG 改成某目录
                    show=args.show)


if __name__ == "__main__":
    main()
