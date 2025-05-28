import time

import serial

ser = serial.Serial('/dev/ttyACM0', 115200 * 32, timeout=0)
t0 = time.time()
cnt = 0
while True:
    data = ser.read(65536)
    cnt += len(data)
    if time.time() - t0 >= 1.0:
        print("Speed: %.2f MB/s" % (cnt / 1024.0 / 1024.0))
        cnt = 0
        t0 = time.time()
