import numpy as np
import pywt
from scipy.signal import find_peaks

from data.data import DataRT


class ArcDetector:
    def __init__(self):
        self.power_mean = -1
        self.pm_lr = 1e-5
        self.wavelet_type = 'sym2'
        self.wavelet_max_level = 1
        self.wavelet_window_size = 256
        self.wavelet_step = 1
        self.wavelet_cache = [0] * self.wavelet_max_level
        self.wavelet_power_bg_cache = [0] * self.wavelet_max_level
        self.wavelet_power_pioneer_cache = [0] * self.wavelet_max_level
        self.db = DataRT(self.wavelet_max_level)
        self.bg_lr = 1e-4
        self.pn_lr = self.bg_lr * 64
        self.wt_th = 2
        self.level_pick = 0
        self.arc_win_smt_cnt = 0
        self.indicator_max_val = 2048
        self.arc_pred_win_shift = 4096
        self.arc_pred_win_s = -1
        self.arc_pred_win_e = -1

    def _af_eval(self):
        idx_s = self.arc_pred_win_s - self.arc_pred_win_shift
        idx_e = self.arc_pred_win_e - self.arc_pred_win_shift
        seq_pick = self.db.db['rt'].seq_power[idx_s:idx_e]
        peaks_idx, _ = find_peaks(seq_pick, height=self.power_mean + 128, distance=128)
        peaks_idx = idx_s + np.array(peaks_idx)
        self.db.db['rt'].info_pred_peaks.extend(peaks_idx.tolist())
        return True

    def infer(self):
        # if self.db.db['rt'].seq_len < self.wavelet_window_size:
        if self.db.db['rt'].seq_len < self.arc_pred_win_shift:
            return
        if self.db.db['rt'].seq_len == self.arc_pred_win_shift:
            self.power_mean = np.mean(np.array(self.db.db['rt'].seq_power[:]))
        _delta = self.db.db['rt'].seq_wt_power_pioneer[-2][self.level_pick] - \
                 self.db.db['rt'].seq_wt_power_bg[-2][self.level_pick]
        self.arc_pred_win_s = \
            self.db.db['rt'].seq_len if _delta > self.wt_th and self.arc_win_smt_cnt == 0 else self.arc_pred_win_s
        self.arc_win_smt_cnt = 16384 if _delta > self.wt_th else self.arc_win_smt_cnt
        self.arc_win_smt_cnt = self.arc_win_smt_cnt - 1 if self.arc_win_smt_cnt > 0 else 0
        self.db.db['rt'].seq_state_pred_arc[-self.arc_pred_win_shift] = self.indicator_max_val \
            if _delta > self.wt_th or self.arc_win_smt_cnt > 0 else 0
        self.db.db['rt'].seq_wavelet[-1] = self.wavelet_cache[:]
        self.db.db['rt'].seq_wt_power_bg[-1] = self.wavelet_power_bg_cache[:]
        self.db.db['rt'].seq_wt_power_pioneer[-1] = self.wavelet_power_pioneer_cache[:]
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        self.arc_pred_win_e = \
            self.db.db['rt'].seq_len if self.arc_pred_win_s > 0 and self.arc_win_smt_cnt == 0 else self.arc_pred_win_e
        if self.arc_pred_win_s > 0 and self.arc_pred_win_e > 0:
            self._af_eval()
            self.arc_pred_win_s, self.arc_pred_win_e = -1, -1
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
        self.power_mean = self.power_mean * (1 - self.pm_lr) + self.db.db['rt'].seq_power[-1] * self.pm_lr
