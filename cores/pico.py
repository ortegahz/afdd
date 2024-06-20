import ctypes
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from picosdk.functions import assert_pico_ok
from picosdk.ps5000a import ps5000a as ps


class PicoBase:
    def __init__(self):
        pass

    def sample(self):
        pass


class Pico5444DMSO(PicoBase):
    def __init__(self):
        super().__init__()
        self.sampleInterval, self.sampleUnits = None, None
        self.maxADC = 0
        self.chandle, self.status = ctypes.c_int16(), dict()
        resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
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
        self.channel_range = ps.PS5000A_RANGE['PS5000A_5V']
        self.status["setChA"] = ps.ps5000aSetChannel(self.chandle,
                                                     ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                     enabled,
                                                     ps.PS5000A_COUPLING['PS5000A_DC'],
                                                     self.channel_range,
                                                     analogue_offset)
        assert_pico_ok(self.status["setChA"])
        self.sizeOfOneBuffer, self.numBuffersToCapture = 512, 8
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

    def _streaming_callback(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.wasCalledBack = True
        destEnd = self.nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        self.nextSample += noOfSamples
        if autoStop:
            self.autoStopOuter = True

    @staticmethod
    def _adc2mV(bufferADC, range, maxADC):
        channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
        vRange = channelInputRanges[range]
        bufferV = [(x * vRange) / maxADC.value for x in bufferADC]
        return bufferV

    def sample(self):
        self.sampleInterval = ctypes.c_int32(256)
        self.sampleUnits = ps.PS5000A_TIME_UNITS['PS5000A_US']
        maxPreTriggerSamples = 0
        autoStopOn = 1
        downsampleRatio = 1
        self.status["runStreaming"] = ps.ps5000aRunStreaming(self.chandle,
                                                             ctypes.byref(self.sampleInterval),
                                                             self.sampleUnits,
                                                             maxPreTriggerSamples,
                                                             self.totalSamples,
                                                             autoStopOn,
                                                             downsampleRatio,
                                                             ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
                                                             self.sizeOfOneBuffer)
        assert_pico_ok(self.status["runStreaming"])
        actualSampleInterval = self.sampleInterval.value
        actualSampleIntervalNs = actualSampleInterval * 1000
        logging.info("Capturing at sample interval %s ns" % actualSampleIntervalNs)
        cFuncPtr = ps.StreamingReadyType(self._streaming_callback)
        while self.nextSample < self.totalSamples and not self.autoStopOuter:
            self.wasCalledBack = False
            self.status["getStreamingLastestValues"] = ps.ps5000aGetStreamingLatestValues(self.chandle, cFuncPtr, None)
            if not self.wasCalledBack:
                time.sleep(0.01)
        logging.info("Done grabbing values.")
        self.maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
        assert_pico_ok(self.status["maximumValue"])
        self.adc2mVChAMax = self._adc2mV(self.bufferCompleteA, self.channel_range, self.maxADC)

    def plot(self):
        time = np.linspace(0, (self.totalSamples - 1) * self.sampleInterval.value * 1000, self.totalSamples)
        # plt.plot(time, self.adc2mVChAMax[:])
        plt.plot(time, self.bufferCompleteA[:])
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.show()

    def close(self):
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
        self.status["close"] = ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])


if __name__ == '__main__':
    pico = Pico5444DMSO()
    pico.sample()
    pico.plot()
    pico.close()
    logging.info(pico)
