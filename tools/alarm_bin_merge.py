# /tools/alarm_bin_merge.py

import glob
import os
import shutil


def collect_and_copy_bins(source_dir, dest_dir):
    """
    查找 source_dir 及其子文件夹下的所有 .bin 文件，
    然后将它们拷贝到 dest_dir 中，并按顺序重命名。
    """
    # --- 1. 检查源目录是否存在 ---
    if not os.path.isdir(source_dir):
        print(f"错误：源目录 '{source_dir}' 不存在。")
        return

    # --- 2. 创建目标目录 (如果不存在) ---
    os.makedirs(dest_dir, exist_ok=True)
    print(f"源目录: {os.path.abspath(source_dir)}")
    print(f"目标目录: {os.path.abspath(dest_dir)}")
    print("-" * 40)

    # --- 3. 使用 glob 递归查找所有 .bin 文件 ---
    # '/**/' 表示匹配所有子目录
    search_pattern = os.path.join(source_dir, '**', '*.bin')
    bin_files = glob.glob(search_pattern, recursive=True)

    total_files = len(bin_files)

    if total_files == 0:
        print("在源目录中没有找到 .bin 文件。")
        return

    # --- 4. 遍历、拷贝和重命名文件 ---
    print(f"找到 {total_files} 个 .bin 文件，开始拷贝...")
    for i, src_path in enumerate(bin_files, 1):
        # 生成新的文件名，例如：0001.bin, 0002.bin ...
        # 使用 zfill 或 f-string 格式化来补零，确保文件名按数字顺序排序
        new_filename = f"{i:04d}.bin"
        dest_path = os.path.join(dest_dir, new_filename)

        try:
            # 使用 shutil.copy2 可以同时复制文件内容和元数据(如修改时间)
            shutil.copy2(src_path, dest_path)
            print(f"  已拷贝: {src_path} -> {dest_path}")
        except Exception as e:
            print(f"  拷贝文件 {src_path} 时出错: {e}")

    # --- 5. 打印最终统计信息 ---
    print("-" * 40)
    print(f"任务完成！成功拷贝 {total_files} 个文件到 '{dest_dir}' 目录。")


if __name__ == "__main__":
    # --- 配置区域 ---
    SOURCE_FOLDER = "/media/manu/ST8000DM004-2U91/tmp/fault-arc/"  # 包含.bin文件的源文件夹
    DESTINATION_FOLDER = "/media/manu/ST8000DM004-2U91/tmp/fault-arc-sorted"  # 存放所有.bin文件的目标文件夹
    # -------------------

    collect_and_copy_bins(SOURCE_FOLDER, DESTINATION_FOLDER)
