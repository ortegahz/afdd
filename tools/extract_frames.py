import os

import cv2

from utils.utils import make_dirs


def extract_frames(ts_file, output_dir, interval=25):
    """
    从 TS 视频文件中按固定间隔抽帧，保存为 BMP 图片。

    :param ts_file: 输入的 TS 文件路径
    :param output_dir: 输出图片保存目录
    :param interval: 抽帧间隔（帧数）
    """
    if not os.path.exists(ts_file):
        raise FileNotFoundError(f"文件不存在: {ts_file}")

    make_dirs(output_dir, reset=True)

    cap = cv2.VideoCapture(ts_file)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {ts_file}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 interval 帧取一帧
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"{saved_count}.bmp")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"已保存 {saved_count} 张图片到 {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 TS 视频文件中按固定间隔抽帧为 BMP 图片")
    parser.add_argument("--ts_file", default="/home/manu/tmp/nir_193398.ts", help="输入的 TS 文件路径")
    parser.add_argument("--output_dir", default="/home/manu/nfs/nir_193398", help="输出图片保存目录")
    parser.add_argument("--interval", type=int, default=5, help="抽帧间隔（默认 25）")

    args = parser.parse_args()

    extract_frames(args.ts_file, args.output_dir, args.interval)
