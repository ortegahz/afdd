#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
截取 /test_data.h5 中每个 dataset 的前 4096 条样本，并保存为
/test_data_first4096.h5
"""

import os

import h5py

SRC_PATH = '/home/manu/tmp/test_data.h5'  # 原文件
DST_PATH = '/home/manu/tmp/test_data_subset.h5'  # 目标文件
N_COPY = 4096  # 截取数量


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
        copy_first_n(fin, fout, N_COPY)

    print(f'Done. Saved first {N_COPY} samples to {DST_PATH}')


if __name__ == '__main__':
    main()
