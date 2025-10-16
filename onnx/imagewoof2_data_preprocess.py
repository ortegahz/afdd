import json
import urllib.request

URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
synset2id = {v[0]: int(k) for k, v in
             json.load(urllib.request.urlopen(URL)).items()}

import pandas as pd, os, shutil, pathlib

CSV = "/home/manu/tmp/imagewoof2-160/noisy_imagewoof.csv"  # 你的 csv
IMG_ROOT = "/home/manu/tmp/imagewoof2-160/"  # csv 中 path 的根目录
OUT_IMG = "/home/manu/tmp/test_images"  # 统一图片目录
OUT_LBL = "/home/manu/tmp/test_labels.txt"

os.makedirs(OUT_IMG, exist_ok=True)
open(OUT_LBL, "w").close()

df = pd.read_csv(CSV, sep=",")  # 如果是逗号用 sep=","

# 1) 只用验证集
df = df[df["is_valid"] == True]

# 2) noisy_labels_0 视为真值
for _, row in df.iterrows():
    syn = row["noisy_labels_0"]
    if syn not in synset2id:  # 可能有不在 1K 类别里的图
        continue
    idx = synset2id[syn]

    src = os.path.join(IMG_ROOT, row["path"])
    dst_name = pathlib.Path(src).name
    dst = os.path.join(OUT_IMG, dst_name)

    # 复制，也可以用 os.symlink(src, dst) 节省空间
    shutil.copy(src, dst)

    with open(OUT_LBL, "a") as f:
        f.write(f"{dst_name} {idx}\n")
