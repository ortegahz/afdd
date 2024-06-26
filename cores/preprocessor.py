import logging

from scipy.signal import *

from data.data import DataPP
from utils.macros import *


class Preprocessor:
    def __init__(self):
        self.db = DataPP()
        self.db_key = 'pp'
        self.sample_rate = SAMPLE_RATE
        self.sub_sample_rate = 400  # 8.93 * 1000 * 1000 / 400 --> 22325.0
        self.sample_rate_new = SAMPLE_RATE / self.sub_sample_rate
        self.filter_cutoff_freq = self.sample_rate_new * 0.4
        self.filter_order = 4
        self.filter_b, self.filter_a, self.filter_zi_org = self._design_highpass_filter()
        self.filter_zi = self.filter_zi_org
        self.seq_power_last_valid = 0

    def reset(self):
        self.seq_power_last_valid = 0
        self.filter_max, self.filter_th, self.filter_th_cnt = -1, -1, 0
        self.filter_zi = self.filter_zi_org
        self.db.reset()

    def _design_highpass_filter(self):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.filter_cutoff_freq / nyq
        b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)
        zi = lfilter_zi(b, a)
        return b, a, zi

    def _realtime_highpass_filter(self, cur_sample):
        y, zo = lfilter(self.filter_b, self.filter_a, [cur_sample], zi=self.filter_zi)
        return y[0], zo

    def process(self):
        logging.info(self.db.db[self.db_key].seq_power[-1])
        filtered_sample, self.filter_zi = self._realtime_highpass_filter(self.db.db[self.db_key].seq_power[-1])
        self.db.db[self.db_key].seq_filtered[-1] = filtered_sample

