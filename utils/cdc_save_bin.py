import signal
import sys
import time
import os
from datetime import datetime

import serial


class SaveData:
    def __init__(self, serial_part="/dev/ttyACM0"):
        self.serial_part = serial_part
        try:
            self.ser = serial.Serial(serial_part, 115200 * 32, timeout=0)
        except serial.SerialException as e:
            self.ser = None
            print(e)
            sys.exit(0)
        self.running = 1
        # 缓冲区设置
        self.buffer_size = 100  # 缓冲100个数据包再写入
        self.data_buffer = bytearray()
        self.read_bytes = 2048

    @staticmethod
    def generate_filename():
        now = datetime.now()
        return now.strftime("%Y%m%d-%H%M%S.bin")

    def run(self):
        t0 = time.time()
        total_bytes = 0

        # 生成基于时间的文件名
        output_filename = os.path.join("/home/manu/tmp", SaveData.generate_filename())
        # 打开输出文件
        output_file = open(output_filename, 'wb', buffering=8388608)  # 使用8MB的文件缓冲区
        print(f"已打开输出文件: {output_filename}")
        while self.running:
            raw_data = self.ser.read(self.read_bytes)
            if raw_data:
                # 直接添加到缓冲区
                self.data_buffer.extend(raw_data)
                # print(len(self.data_buffer))
                total_bytes += len(raw_data)
                # 如果缓冲区足够大，写入文件
                if len(self.data_buffer) >= self.buffer_size * self.read_bytes:
                    output_file.write(self.data_buffer)
                    self.data_buffer = bytearray()  # 清空缓冲区
                if time.time() - t0 >= 1.0:
                    print("Speed: %.2f MB/s" % (total_bytes / 1024.0 / 1024.0))
                    total_bytes = 0
                    t0 = time.time()

        output_file.close()
        print("{}文件保存成功".format(output_filename))

    def signal_handler(self, *args):
        self.running = False
        print("Ctrl+C 被你按下了！")

    def __del__(self):
        self.running = 1
        if self.ser:
            print("{}串口关闭".format(self.serial_part))
            self.ser.close()
        print("程序释放")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("端口号未输入，请重试！")
    #     sys.exit(0)
    # s = SaveData(sys.argv[1])
    s = SaveData()
    signal.signal(signal.SIGINT, s.signal_handler)
    s.run()
