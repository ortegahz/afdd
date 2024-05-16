import numpy as np
import pywt
from scipy.signal import find_peaks

from data.data import DataRT


class ArcDetector:
    def __init__(self):
        self.power_mean = -1
        self.pm_lr = 1e-4
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
        self.peak_eval_win = list()
        self.peak_eval_win_size = 128
        self.af_win_size = 256
        self.last_peak_idx = -1

    def _af_eval(self):
        idx_s = self.arc_pred_win_s - self.arc_pred_win_shift
        idx_e = self.arc_pred_win_e - self.arc_pred_win_shift
        seq_pick = self.db.db['rt'].seq_power[idx_s:idx_e]
        peaks_idx, _ = find_peaks(seq_pick, height=self.power_mean + 128, distance=128)
        af_cnt = 0
        for i in range(len(peaks_idx) - 1):
            peak_s, peak_e = peaks_idx[i], peaks_idx[i + 1]
            seq_pick_t = np.array(seq_pick[peak_s:peak_e]).astype(float)
            seq_pick_t_delta = np.abs(seq_pick_t - self.power_mean)
            af_score = len(seq_pick_t_delta[seq_pick_t_delta < 4])
            self.db.db['rt'].info_af_scores.append(af_score)
            af_cnt = af_cnt + 1 if af_score > 16 else 0
        self.db.db['rt'].info_af_scores.append(0)
        peaks_idx = idx_s + np.array(peaks_idx)
        self.db.db['rt'].info_pred_peaks.extend(peaks_idx.tolist())
        return af_cnt > 4

    def infer_v0(self):
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
        self.db.db['rt'].seq_state_pred_arc[-self.arc_pred_win_shift] = self.indicator_max_val / 2 \
            if _delta > self.wt_th or self.arc_win_smt_cnt > 0 else 0
        self.db.db['rt'].seq_wavelet[-1] = self.wavelet_cache[:]
        self.db.db['rt'].seq_wt_power_bg[-1] = self.wavelet_power_bg_cache[:]
        self.db.db['rt'].seq_wt_power_pioneer[-1] = self.wavelet_power_pioneer_cache[:]
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        self.arc_pred_win_e = \
            self.db.db['rt'].seq_len if self.arc_pred_win_s > 0 and self.arc_win_smt_cnt == 0 else self.arc_pred_win_e
        if self.arc_pred_win_s > 0 and self.arc_pred_win_e > 0:
            if self._af_eval():
                self.db.db['rt'].seq_state_pred_arc[self.arc_pred_win_s] = self.indicator_max_val
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

    def _detect_peak(self, cur_val, win_size=256, peak_th=0):
        self.peak_eval_win.append(cur_val)
        self.peak_eval_win = self.peak_eval_win[-win_size:]
        if len(self.peak_eval_win) < win_size:
            return -1
        peak_candidate_idx = win_size // 2
        if self.peak_eval_win[peak_candidate_idx] == max(self.peak_eval_win) and \
                self.peak_eval_win[peak_candidate_idx] > peak_th:
            self.peak_eval_win.clear()
            return peak_candidate_idx
        return -1

    def infer_v1(self):
        if self.db.db['rt'].seq_len < self.af_win_size * 2:
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size, peak_th=self.power_mean + 128)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:
            return
        _seq_pick = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick_delta = np.abs(_seq_pick - self.power_mean)
        # _delta_th = (self.db.db['rt'].seq_power[peak_idx] - self.power_mean) / 16.
        _delta_th = 16
        _delta_score = len(_seq_pick_delta[_seq_pick_delta < _delta_th])
        _af_score = _delta_score if peak_idx - self.last_peak_idx < self.af_win_size * 2 else 0
        self.db.db['rt'].info_af_scores.append(_af_score)
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        if _af_score > 16 or self.db.db['rt'].seq_power[peak_idx] > 4096 - 128:
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val] * self.af_win_size
        self.last_peak_idx = peak_idx
