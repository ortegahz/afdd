import logging

import pywt

from data.data import DataRT


class ArcDetector:
    def __init__(self):
        self.wavelet_type = 'sym2'
        self.wavelet_max_level = 1
        self.wavelet_window_size = 256
        self.wavelet_step = 1
        self.wavelet_cache = [0] * self.wavelet_max_level
        self.wavelet_power_bg_cache = [0] * self.wavelet_max_level
        self.wavelet_power_pioneer_cache = [0] * self.wavelet_max_level
        self.db = DataRT(self.wavelet_max_level)
        self.bg_lr = 1e-4
        self.pn_lr = self.bg_lr * 10
        self.wt_th = 2
        self.level_pick = 0

    def infer(self):
        if self.db.db['rt'].seq_len < self.wavelet_window_size:
            return
        _delta = self.db.db['rt'].seq_wt_power_pioneer[-2][self.level_pick] - \
                 self.db.db['rt'].seq_wt_power_bg[-2][self.level_pick]
        # logging.info(f'delta: {_delta}')
        self.db.db['rt'].seq_state_pred_arc[-2] = 2048 if _delta > self.wt_th else 0
        self.db.db['rt'].seq_wavelet[-1] = self.wavelet_cache[:]
        self.db.db['rt'].seq_wt_power_bg[-1] = self.wavelet_power_bg_cache[:]
        self.db.db['rt'].seq_wt_power_pioneer[-1] = self.wavelet_power_pioneer_cache[:]
        if self.db.db['rt'].seq_len % self.wavelet_step != 0:
            return
        seq_power_pick = self.db.db['rt'].seq_power[-self.wavelet_window_size:]
        coeffs = pywt.wavedec(seq_power_pick, self.wavelet_type, level=self.wavelet_max_level)
        for level, coeff in enumerate(coeffs[1:], start=1):
            self.wavelet_cache[level - 1] = max(coeff)
            self.wavelet_power_bg_cache[level - 1] = \
                self.wavelet_power_bg_cache[level - 1] * (1 - self.bg_lr) + max(coeff) * self.bg_lr
            self.wavelet_power_pioneer_cache[level - 1] = \
                self.wavelet_power_pioneer_cache[level - 1] * (1 - self.pn_lr) + max(coeff) * self.pn_lr
