#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
截取 /test_data.h5 中每个 dataset 的前 4096 条样本，并保存为
/test_data_first4096.h5
"""

import os

SRC_PATH = '/home/manu/tmp/test_data.h5'  # 原文件
DST_PATH = '/home/manu/tmp/test_data_subset.h5'  # 目标文件
# N_COPY = 32
N_COPY = 60000
SEED = 128

import numpy as np
import h5py


def copy_balanced_n(src_group: h5py.Group,
                    dst_group: h5py.Group,
                    n_total: int = 4096,
                    label_key: str = 'labels',
                    seed: int = 128) -> tuple:
    """
    递归复制 group/dataset；在含 `label_key` 的叶子 group 内，
    对 label=0/1 均衡抽样，各取 min(pos, neg, n_total//2) 条。
    复制完成后返回 (pos_cnt, neg_cnt)，方便统计。

    返回值
    ----
    (pos_cnt, neg_cnt) : tuple[int, int]
        本 group（含其子 group）最终被写入 dst_group 的
        正 / 负 样本总数。
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1. 判断当前 group 是否叶子
    # ------------------------------------------------------------
    is_leaf = (label_key in src_group and
               isinstance(src_group[label_key], h5py.Dataset))

    pos_total = 0  # 本 group（含递归子树）最终写入的正样本数
    neg_total = 0  # 本 group（含递归子树）最终写入的负样本数

    # ------------------------------------------------------------
    # 2. 叶子 group：做均衡采样
    # ------------------------------------------------------------
    if is_leaf:
        labels = src_group[label_key][:]
        if labels.ndim != 1:
            raise ValueError(f'{src_group[label_key].name} 必须是一维标签')
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f'{src_group[label_key].name} 仅允许 0 / 1 标签')

        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        if pos_idx.size == 0 or neg_idx.size == 0:
            raise RuntimeError(f'{src_group.name} 标签只有单边，无法均衡采样')

        n_each = min(pos_idx.size, neg_idx.size, n_total // 2)
        sel_idx = np.concatenate([
            rng.choice(pos_idx, n_each, replace=False),
            rng.choice(neg_idx, n_each, replace=False)
        ])
        rng.shuffle(sel_idx)
        sel_sorted = np.sort(sel_idx)
        order = np.argsort(np.searchsorted(sel_sorted, sel_idx))

        # 复制该 group 内所有 dataset
        for dname, dset_in in src_group.items():
            if not isinstance(dset_in, h5py.Dataset):
                continue
            data_sorted = dset_in[sel_sorted]
            data = data_sorted[order]

            dset_out = dst_group.create_dataset(
                dname, data=data, compression=dset_in.compression)
            for k, v in dset_in.attrs.items():
                dset_out.attrs[k] = v

        # 打印当前叶子 group 的采样情况
        print(f'[leaf] {src_group.name or "/"} -> pos={n_each}, neg={n_each}')

        pos_total += n_each
        neg_total += n_each

    # ------------------------------------------------------------
    # 3. 非叶子 group
    # ------------------------------------------------------------
    else:
        for name, item in src_group.items():
            if isinstance(item, h5py.Dataset):
                # 与 copy_first_n 一致：仅复制前 n_total 条
                length = item.shape[0]
                slice_len = min(length, n_total)
                data = item[:slice_len]
                dset = dst_group.create_dataset(
                    name, data=data, compression=item.compression)
                for k, v in item.attrs.items():
                    dset.attrs[k] = v
            elif isinstance(item, h5py.Group):
                sub_dst = dst_group.create_group(name)
                for k, v in item.attrs.items():
                    sub_dst.attrs[k] = v
                # 递归并累加统计
                p_sub, n_sub = copy_balanced_n(item, sub_dst,
                                               n_total=n_total,
                                               label_key=label_key,
                                               seed=seed)
                pos_total += p_sub
                neg_total += n_sub
            else:
                print(f'Unknown item type: {name} -> {type(item)}')

    # 复制当前 group 属性
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v

    return pos_total, neg_total


def copy_random_n(src_group: h5py.Group,
                  dst_group: h5py.Group,
                  n: int = 4096,
                  seed: int = 128):
    rng = np.random.default_rng(seed)

    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            length = item.shape[0]
            sample_len = min(length, n)

            idx_rand = rng.choice(length, size=sample_len, replace=False)
            idx_sorted = np.sort(idx_rand)  # 先排好序
            data_sorted = item[idx_sorted]  # 读取
            # 再复原到随机顺序
            order = np.argsort(np.searchsorted(idx_sorted, idx_rand))
            data = data_sorted[order]

            dset = dst_group.create_dataset(
                name, data=data,
                compression=item.compression
            )
            for k, v in item.attrs.items():
                dset.attrs[k] = v

        elif isinstance(item, h5py.Group):
            sub = dst_group.create_group(name)
            for k, v in item.attrs.items():
                sub.attrs[k] = v
            copy_random_n(item, sub, n, seed)


def copy_first_n(src_group: h5py.Group,
                 dst_group: h5py.Group,
                 n: int = 4096):
    """
    递归复制 group / dataset
    """
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            # 取第 0 维的前 n 个；不足 n 时全部复制
            length = item.shape[0]
            slice_len = min(length, n)
            data = item[:slice_len]  # 读取到内存
            dset = dst_group.create_dataset(
                name,
                data=data,
                compression=item.compression)  # 如果原来用了压缩也保持
            # 复制 dataset 属性
            for k, v in item.attrs.items():
                dset.attrs[k] = v
        elif isinstance(item, h5py.Group):
            # 创建子 group，递归
            sub_group = dst_group.create_group(name)
            # 复制 group 属性
            for k, v in item.attrs.items():
                sub_group.attrs[k] = v
            copy_first_n(item, sub_group, n)
        else:
            print(f'Unknown item type: {name} -> {type(item)}')


def main():
    if not os.path.isfile(SRC_PATH):
        raise FileNotFoundError(SRC_PATH)

    # 若目标文件已存在先删除
    if os.path.exists(DST_PATH):
        os.remove(DST_PATH)

    with h5py.File(SRC_PATH, 'r') as fin, \
            h5py.File(DST_PATH, 'w') as fout:

        # 复制文件级别（root）的属性
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # 递归复制内容
        # copy_first_n(fin, fout, N_COPY)
        copy_random_n(fin, fout, N_COPY, seed=SEED)
        # pos_cnt, neg_cnt = copy_balanced_n(fin, fout,
        #                                    n_total=N_COPY,
        #                                    label_key='labels',
        #                                    seed=SEED)
        #
        # print('====================================')
        # print(f'全部采样完成：正样本 {pos_cnt} 条，负样本 {neg_cnt} 条'
        #       f'（比例 {pos_cnt / (pos_cnt + neg_cnt):.3f}）')
        # print('保存至 ->', DST_PATH)
        print(f'Done. Saved first {N_COPY} samples to {DST_PATH}')


if __name__ == '__main__':
    main()
