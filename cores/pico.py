import ctypes
import logging
from multiprocessing import Process, Queue, Event

import matplotlib.pyplot as plt
import numpy as np
from picosdk.functions import assert_pico_ok
from picosdk.ps5000a import ps5000a as ps

from cores.arc_detector import ArcDetector
from utils.utils import set_logging


class PicoBase:
    def __init__(self):
        pass

    def sample(self):
        pass


class Pico5444DMSO(PicoBase):
    def __init__(self, data_queue, overflow_queue, stop_event):
        super().__init__()
        self.sampleInterval, self.sampleUnits = -1, -1
        self.maxADC = ctypes.c_int16()
        self.chandle, self.status = ctypes.c_int16(), dict()
        resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_14BIT"]
        self.status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, resolution)
        try:
            assert_pico_ok(self.status["openunit"])
        except:
            powerStatus = self.status["openunit"]
            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            elif powerStatus == 282:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise
            assert_pico_ok(self.status["changePowerSource"])
        enabled = 1
        analogue_offset = 0.0
        self.channel_range = ps.PS5000A_RANGE['PS5000A_10V']  # Assuming the sensor output range is -5V to 5V
        self.status["setChA"] = ps.ps5000aSetChannel(self.chandle,
                                                     ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                     enabled,
                                                     ps.PS5000A_COUPLING['PS5000A_DC'],
                                                     self.channel_range,
                                                     analogue_offset)
        assert_pico_ok(self.status["setChA"])
        self.sizeOfOneBuffer, self.numBuffersToCapture = 512, 1
        self.totalSamples = self.sizeOfOneBuffer * self.numBuffersToCapture
        self.bufferCompleteA = np.zeros(shape=self.totalSamples, dtype=np.int16)
        self.bufferAMax = np.zeros(shape=self.sizeOfOneBuffer, dtype=np.int16)
        memory_segment = 0
        self.status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(self.chandle,
                                                                  ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                                  self.bufferAMax.ctypes.data_as(
                                                                      ctypes.POINTER(ctypes.c_int16)),
                                                                  None,
                                                                  self.sizeOfOneBuffer,
                                                                  memory_segment,
                                                                  ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'])
        assert_pico_ok(self.status["setDataBuffersA"])
        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False
        self.data_queue = data_queue
        self.overflow_queue = overflow_queue
        self.stop_event = stop_event

        # Get the maximum ADC value
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
        assert_pico_ok(self.status["maximumValue"])

    def _streaming_callback(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.wasCalledBack = True
        self.data_queue.put(self.bufferAMax.copy())
        # destEnd = self.nextSample + noOfSamples
        # sourceEnd = startIndex + noOfSamples
        # self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        # self.nextSample += noOfSamples
        # if overflow:
        #     self.overflow_queue.put(True)
        # if autoStop:
        #     self.autoStopOuter = True

    def sample(self):
        self.sampleInterval = ctypes.c_int32(1)
        self.sampleUnits = ps.PS5000A_TIME_UNITS['PS5000A_US']
        maxPreTriggerSamples = 0
        autoStopOn = 0  # Disable auto stop
        downsampleRatio = 1
        self.status["runStreaming"] = ps.ps5000aRunStreaming(self.chandle,
                                                             ctypes.byref(self.sampleInterval),
                                                             self.sampleUnits,
                                                             maxPreTriggerSamples,
                                                             0,  # Continuous streaming
                                                             autoStopOn,
                                                             downsampleRatio,
                                                             ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
                                                             self.sizeOfOneBuffer)
        assert_pico_ok(self.status["runStreaming"])
        actualSampleInterval = self.sampleInterval.value
        actualSampleIntervalNs = actualSampleInterval * 1000
        logging.info("Capturing at sample interval %s ns" % actualSampleIntervalNs)
        cFuncPtr = ps.StreamingReadyType(self._streaming_callback)

        self.data_queue.put(self.maxADC.value)
        self.data_queue.put(self.channel_range)
        self.data_queue.put(self.maxADC)

        while not self.stop_event.is_set():
            self.wasCalledBack = False
            self.status["getStreamingLastestValues"] = ps.ps5000aGetStreamingLatestValues(self.chandle, cFuncPtr, None)
            if self.wasCalledBack:
                # self.data_queue.put(self.bufferCompleteA[:self.nextSample].copy())
                self.nextSample = 0
            # time.sleep(0.01)

    def close(self):
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
        self.status["close"] = ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])


def adc2V(bufferADC, range, maxADC):
    channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    vRange = channelInputRanges[range]
    # bufferV = [(x * vRange) / 100 / maxADC.value for x in bufferADC]
    bufferV = bufferADC * float(vRange) / 100. / float(maxADC.value)
    return bufferV


def data_acquisition_process(data_queue, overflow_queue, stop_event):
    pico = Pico5444DMSO(data_queue, overflow_queue, stop_event)
    try:
        pico.sample()
    finally:
        pico.close()


def data_plotting_process(data_queue, overflow_queue, stop_event):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def on_key(event):
        if event.key == 'q':
            stop_event.set()

    fig.canvas.mpl_connect('key_press_event', on_key)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    ax.set_xlim(0, 2 * 1e6)
    ax.set_ylim(-5, 5)

    sample_interval = 256  # 256 us
    num_samples = int(2 * 1e6 / sample_interval)
    data_buffer = np.zeros(num_samples)
    max_adc_value = data_queue.get()
    channel_range = data_queue.get()
    maxADC = data_queue.get()

    while not stop_event.is_set():
        if not overflow_queue.empty():
            overflow = overflow_queue.get()
            if overflow:
                logging.warning("Data overflow detected! Samples may be lost.")

        if not data_queue.empty():
            new_data = data_queue.get()
            logging.info((data_queue.qsize(), len(new_data)))
            current_values = adc2V(new_data, channel_range, maxADC)
            # logging.info(current_values[:4])
            # data_buffer = np.roll(data_buffer, -len(new_data))
            # data_buffer[-len(new_data):] = current_values
            # time_array = np.linspace(0, (len(data_buffer) - 1) * sample_interval, len(data_buffer))
            # line.set_xdata(time_array)
            # line.set_ydata(data_buffer)
            # ax.relim()
            # ax.autoscale_view()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
        # time.sleep(0.01)

    plt.ioff()
    plt.show()


def afdd_process(data_queue, overflow_queue, stop_event):
    # set_logging()
    arc_detector = ArcDetector()

    max_adc_value = data_queue.get()
    channel_range = data_queue.get()
    maxADC = data_queue.get()

    try:
        while not stop_event.is_set():
            if not overflow_queue.empty():
                overflow = overflow_queue.get()
                if overflow:
                    logging.warning("Data overflow detected! Samples may be lost.")

            if not data_queue.empty():
                new_data = data_queue.get()
                logging.info((data_queue.qsize(), len(new_data)))
                current_values = adc2V(new_data, channel_range, maxADC)
                for idx in range(0, 512, arc_detector.sub_sample_rate):  # 512 should be same as pre-set buffer len
                    cur_power = current_values[idx] * 2048 / 40 + 2048
                    arc_detector.db.update(
                        cur_power=cur_power,
                        cur_hf=1.0,
                        cur_state_gt_arc=0.0,
                        cur_state_gt_normal=0.0)
                    arc_detector.infer_v2()
    finally:
        logging.info("Saving data and exiting...")
        np.save(r'C:\Users\admin\Desktop\manu\seq_power.npy',
                (np.array(arc_detector.db.db['rt'].seq_power) - 2048) * 40 / 2048
                )


if __name__ == '__main__':
    set_logging()
    data_queue = Queue()
    overflow_queue = Queue()
    stop_event = Event()
    acquisition_process = Process(target=data_acquisition_process, args=(data_queue, overflow_queue, stop_event))
    # _process = Process(target=data_plotting_process, args=(data_queue, overflow_queue, stop_event))
    _process = Process(target=afdd_process, args=(data_queue, overflow_queue, stop_event))
    acquisition_process.start()
    _process.start()
    acquisition_process.join()
    _process.join()
