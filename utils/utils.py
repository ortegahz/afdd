import logging
import os
import shutil


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def make_dirs(dir_root, reset=False):
    if dir_root is None:
        return
    if os.path.exists(dir_root) and reset:
        shutil.rmtree(dir_root)
    os.makedirs(os.path.join(dir_root), exist_ok=True)


def load_bin(path, int_size=2):
    int_list = list()
    with open(path, 'rb') as file:
        while True:
            bytes_read = file.read(int_size)
            if not bytes_read:
                break
            integer = int.from_bytes(bytes_read, 'little')
            int_list.append(integer)
    return int_list
