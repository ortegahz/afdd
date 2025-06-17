# split_and_save.py
from pathlib import Path

import numpy as np


def read_groups(txt_file, sep_char='*'):
    """
    把 txt_file 按由 sep_char 组成的一整行切分成多组，
    返回一个 ndarray 列表。
    """
    groups = []  # 最终结果：存放 np.array 的列表
    current_buffer = []  # 当前正在累积的一组数值

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()

            # 1. 判定是否为分隔行（全由 sep_char 组成，且非空）
            if s and set(s) == {sep_char}:
                if current_buffer:  # 把上一组收尾
                    groups.append(np.array(current_buffer, dtype=float))
                    current_buffer = []  # 重新开始收集下一组
                continue  # 跳过分隔行

            # 2. 普通数字行
            if s:  # 跳过空行
                try:
                    current_buffer.append(float(s))
                except ValueError:  # 行里不是合法数字就忽略
                    print(f"Warning: 跳过无法解析的行 -> {s!r}")

    # 文件结束时，别忘了把最后一组也存下来
    if current_buffer:
        groups.append(np.array(current_buffer, dtype=float))

    return groups


def save_groups(groups, out_prefix='signal_', out_dir='output', fmt='%.6f'):
    """
    将 groups（np.array 列表）依次保存成独立文件。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, arr in enumerate(groups, 1):
        fname = out_path / f"{out_prefix}{idx}.txt"
        np.savetxt(fname, arr, fmt=fmt)
        print(f"[OK] 写出 {fname}  ({len(arr)} 个采样点)")


if __name__ == '__main__':
    INPUT_FILE = 'data.txt'  # 原始文件名
    groups = read_groups(INPUT_FILE)
    print(f"共解析出 {len(groups)} 组信号")
    save_groups(groups)
