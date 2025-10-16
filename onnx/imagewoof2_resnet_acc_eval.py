import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# 跟量化时保持一致的预处理
MEAN = np.array([123.68, 116.78, 103.94], dtype=np.float32)  # BGR
H, W = 224, 224  # resnet50 默认输入


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((W, H))
    img = np.asarray(img, dtype=np.float32)
    img = img - MEAN
    # (H,W,C) -> (1,C,H,W)
    img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
    return img


def evaluate(model_path, img_folder, label_file):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    y_pred, y_true = [], []

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        fname, label = line.strip().split()
        img_path = os.path.join(img_folder, fname)

        inp = preprocess(img_path)
        logits = sess.run(None, {input_name: inp})[0]  # (1, num_classes)
        pred = int(np.argmax(logits, axis=1)[0])

        y_pred.append(pred)
        y_true.append(int(label))

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return acc, f1


input_model_path = "/home/manu/tmp/resnetv2_50_Opset18.onnx"
output_model_path = "/home/manu/tmp/resnetv2_50_Opset18-quant.onnx"
test_img_dir = "/home/manu/tmp/test_images"
test_label_file = "/home/manu/tmp/test_labels.txt"

print("Evaluating FP32 model ...")
acc_fp32, f1_fp32 = evaluate(input_model_path, test_img_dir, test_label_file)
print(f"FP32  ->  ACC = {acc_fp32:.4f},  F1 = {f1_fp32:.4f}")

print("Evaluating INT8 model ...")
acc_int8, f1_int8 = evaluate(output_model_path, test_img_dir, test_label_file)
print(f"INT8  ->  ACC = {acc_int8:.4f},  F1 = {f1_int8:.4f}")
