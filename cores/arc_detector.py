import logging
import os
import pickle
import sys
from subprocess import *

import numpy as np
import pywt
from scipy.signal import *

from data.data import DataRT
from utils.macros import *


class ArcDetector:
    def __init__(self):
        with open('/home/manu/tmp/model.pickle', 'rb') as f:
            self.classifier = pickle.load(f)
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
        self.indicator_max_val = 4096
        self.arc_pred_win_shift = 4096
        self.arc_pred_win_s = -1
        self.arc_pred_win_e = -1
        self.peak_eval_win = list()
        self.peak_eval_win_size = 128
        self.af_win_size = int(SAMPLE_RATE / 50)
        self.last_peak_idx = -1
        self.sample_rate = SAMPLE_RATE  # 256 * 50
        self.filter_cutoff_freq = 256  # hz
        self.filter_order = 4
        self.filter_b, self.filter_a, self.filter_zi_org = self._design_highpass_filter()
        self.filter_zi = self.filter_zi_org
        self.sample_win_size = self.sample_rate  # 1s
        self.sample_cnt = 1024
        self.samples_neg, self.samples_pos = list(), list()
        self.seq_template = None
        self.seq_template_idx = -1
        self.filter_max, self.filter_th, self.filter_th_cnt = -1, -1, 0
        self.alarm_arc_cnt = 0
        self.alarm_arc_th = 1400 / self.indicator_max_val
        self.alarm_arc_descend_peak_cnt = 0
        self.alarm_arc_idx_s = -1
        self.alarm_arc_idx_e = -1
        self.alarm_arc_descend_exit = False
        self.alarm_arc_state = 0
        self.peak_miss_cnt = 0
        self.peak_bulge_mask_cnt = 0
        self.peak_bulge_anchor_idx = -1
        self.peak_bulge_cnt = 0

    @staticmethod
    def _update_svm_label_file(seq_pick, path_out='/home/manu/tmp/smartsd', subset='neg'):
        idx_feat = 0
        with open(path_out, 'a') as f:
            label = '+1' if 'pos' in subset else '-1'
            f.write(label + ' ')
            for feat in seq_pick:
                f.write(f'{idx_feat + 1}:{feat} ')
                idx_feat += 1
        with open(path_out, 'a') as f:
            f.write('\n')

    def save_samples(self, path_save='/home/manu/tmp/smartsd'):
        # if len(self.samples_neg) > len(self.samples_pos):
        #     self.samples_neg = random.sample(self.samples_neg, len(self.samples_pos))
        # logging.info(f'len self.samples_pos -> {len(self.samples_pos)}')
        # logging.info(f'len self.samples_neg -> {len(self.samples_neg)}')
        for seq_pick in self.samples_pos:
            self._update_svm_label_file(seq_pick, path_out=path_save, subset='pos')
        for seq_pick in self.samples_neg:
            self._update_svm_label_file(seq_pick, path_out=path_save, subset='neg')

    def reset(self):
        self.peak_bulge_cnt = 0
        self.peak_bulge_anchor_idx = -1
        self.peak_bulge_mask_cnt = 0
        self.peak_miss_cnt = 0
        self.alarm_arc_state = 0
        self.alarm_arc_idx_e = -1
        self.alarm_arc_descend_exit = False
        self.alarm_arc_idx_s = -1
        self.alarm_arc_descend_peak_cnt = 0
        self.last_peak_idx = -1
        self.peak_eval_win.clear()
        self.alarm_arc_cnt = 0
        self.filter_max, self.filter_th, self.filter_th_cnt = -1, -1, 0
        self.samples_neg.clear()
        self.samples_pos.clear()
        self.filter_zi = self.filter_zi_org
        self.db.reset()

    def _design_highpass_filter(self):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.filter_cutoff_freq / nyq
        # b, a = butter(self.filter_order, normal_cutoff, btype='high', analog=False)
        # rp = 1
        # b, a = cheby1(self.filter_order, rp, normal_cutoff, btype='high', analog=False)
        rs = 64
        b, a = cheby2(self.filter_order, rs, normal_cutoff, btype='high', analog=False)
        zi = lfilter_zi(b, a)
        return b, a, zi

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

    def _svm_infer(self, seq, suffix='', path_label='./rtsvm', dir_libsvm='/home/manu/nfs/libsvm'):
        is_win32 = (sys.platform == 'win32')
        if is_win32:
            svmscale_exe = os.path.join(dir_libsvm, 'windows', 'svm-scale.exe')
            svmpredict_exe = os.path.join(dir_libsvm, 'windows', 'svm-predict.exe')
        else:
            svmscale_exe = os.path.join(dir_libsvm, 'svm-scale')
            svmpredict_exe = os.path.join(dir_libsvm, 'svm-predict')
        # range_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.range')
        # model_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.model')
        range_file = os.path.join(dir_libsvm, 'tools', 'smartsd' + suffix + '.range')
        model_file = os.path.join(dir_libsvm, 'tools', 'smartsd' + suffix + '.model')
        test_pathname = path_label
        scaled_test_file = path_label + '.scale'
        predict_test_file = path_label + '.predict'
        if os.path.exists(path_label):
            os.remove(path_label)
        self._update_svm_label_file(seq, path_label)
        cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        cmd = '{0} -b 0 "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        Popen(cmd, shell=True).communicate()
        with open(predict_test_file) as f:
            lines = f.readlines()
        # return int(lines[1].split(' ')[0])
        return int(lines[0].strip())

    def _realtime_highpass_filter(self, cur_sample):
        y, zo = lfilter(self.filter_b, self.filter_a, [cur_sample], zi=self.filter_zi)
        return y[0], zo

    def infer_v1(self):
        filtered_sample, self.filter_zi = self._realtime_highpass_filter(self.db.db['rt'].seq_power[-1])
        self.db.db['rt'].seq_filtered[-1] = filtered_sample
        if self.db.db['rt'].seq_len < self.af_win_size * 2:  # waiting for _seq_pick_last
            return
        # self.filter_max = filtered_sample if self.filter_max < filtered_sample else self.filter_max
        # self.filter_th_cnt = self.filter_th_cnt + 1 if filtered_sample > self.filter_th else self.filter_th_cnt
        # if filtered_sample > self.filter_th and self.filter_th_cnt > SAMPLE_RATE:
        #     self.filter_th, self.filter_th_cnt = self.filter_max, 0
        # self.db.db['rt'].seq_filter_envelope[-1] = self.filter_th
        # self.sample_cnt = self.sample_win_size if filtered_sample > 2 else self.sample_cnt
        # self.sample_cnt = self.sample_cnt - 1 if self.sample_cnt > 0 else self.sample_cnt
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size, peak_th=self.power_mean + 8)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:
            return
        _seq_pick = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick_delta = np.abs(_seq_pick - self.power_mean)
        # _delta_th = (self.db.db['rt'].seq_power[peak_idx] - self.power_mean) / 16.
        _delta_th = (self.db.db['rt'].seq_power[peak_idx] - self.power_mean) * 0.016
        _delta_score = len(_seq_pick_delta[_seq_pick_delta < _delta_th])
        _af_score = _delta_score if peak_idx - self.last_peak_idx < self.af_win_size * 2 else 0
        # logging.info(peak_idx - self.seq_template_idx)
        # _af_score = np.mean(np.abs(_seq_pick - self.seq_template)) if self.seq_template is not None else 0
        # self.seq_template, self.seq_template_idx = _seq_pick, peak_idx
        self.db.db['rt'].info_af_scores.append(_af_score)
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        # if _af_score > 16 or self.db.db['rt'].seq_power[peak_idx] > 4096 - 128:
        state_gt_arc = self.db.db['rt'].seq_state_gt_arc[-1]
        if self.sample_cnt > 0:
            # _svm_score = self._svm_infer(_seq_pick)
            _data = _seq_pick[np.newaxis, :]
            _score = self.classifier.infer(_data)[0]
            self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
                [_score * self.indicator_max_val] * self.af_win_size
            if state_gt_arc > 0:
                self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                    [self.indicator_max_val / 2] * self.af_win_size
                self.samples_pos.append(_seq_pick)
            else:
                self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                    [self.indicator_max_val / 4] * self.af_win_size
                self.samples_neg.append(_seq_pick)
        self.last_peak_idx = peak_idx

    def infer_v2(self):
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        self.alarm_arc_idx_e = self.last_peak_idx if self.peak_miss_cnt > self.af_win_size * 4 else self.alarm_arc_idx_e
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            logging.info(f'self.peak_bulge_mask_cnt --> {self.peak_bulge_mask_cnt}')
            _delta_peak_th = 256 if self.peak_bulge_mask_cnt > 0 else 2048
            _delta_peak = \
                self.db.db['rt'].seq_power[self.alarm_arc_idx_s] - self.db.db['rt'].seq_power[self.alarm_arc_idx_e]
            logging.info(f'_delta_peak --> {_delta_peak}')
            self.alarm_arc_cnt = self.alarm_arc_cnt - 16 if _delta_peak > _delta_peak_th else self.alarm_arc_cnt
            _alarm_arc_end_idx = self.last_peak_idx
            _idx_e = self.alarm_arc_idx_e + self.af_win_size * 4
            _idx_e = _idx_e if _idx_e < self.db.db['rt'].seq_len else self.db.db['rt'].seq_len
            _seq_pick_hf_alarm = \
                np.array(
                    self.db.db['rt'].seq_hf[self.alarm_arc_idx_s - self.af_win_size:_idx_e]).astype(float)
            _hf_cnt = np.sum(_seq_pick_hf_alarm > 0)
            logging.info(f'_delta_peak_th --> {_delta_peak_th}')
            logging.info(f'_hf_cnt --> {_hf_cnt}')
            logging.info(f'self.alarm_arc_cnt --> {self.alarm_arc_cnt}')
            logging.info(f'self.alarm_arc_idx_s --> {self.alarm_arc_idx_s}')
            logging.info(f'self.last_peak_idx --> {self.last_peak_idx}')
            if _hf_cnt > 0 and self.alarm_arc_cnt > 4:
                logging.info(f'alarm idx --> {self.last_peak_idx}')
                self.db.db['rt'].seq_state_pred_arc[self.last_peak_idx - self.af_win_size:self.last_peak_idx] = \
                    [self.indicator_max_val] * self.af_win_size
                self.alarm_arc_state = 0
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size, peak_th=self.power_mean + 8)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:  # seq filter
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.peak_bulge_mask_cnt = \
                self.peak_bulge_mask_cnt - 1 if self.peak_bulge_mask_cnt > 0 else self.peak_bulge_mask_cnt
            # self.alarm_arc_cnt = max(0, self.alarm_arc_cnt)
            self.peak_miss_cnt += 1
            return
        self.peak_miss_cnt = 0
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score = self.classifier.infer(_data)[0]
        self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
            [_score * self.indicator_max_val] * self.af_win_size
        _delta_peak = self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.last_peak_idx]
        _bulge_th = 512
        if _delta_peak > _bulge_th and self.peak_bulge_anchor_idx < 0 and self.last_peak_idx > 0:
            self.peak_bulge_anchor_idx = self.last_peak_idx
            self.peak_bulge_cnt += 1
            logging.info(f'_delta_peak -- > {_delta_peak}')
            logging.info(f'int self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        if self.peak_bulge_anchor_idx > 0:
            _delta_peak_bulge = \
                self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.peak_bulge_anchor_idx]
            self.peak_bulge_cnt = self.peak_bulge_cnt + 1 if _delta_peak_bulge > _bulge_th else self.peak_bulge_cnt
            logging.info(f'self.peak_bulge_cnt --> {self.peak_bulge_cnt}')
        self.peak_bulge_mask_cnt = SAMPLE_RATE if self.peak_bulge_cnt > 2 else self.peak_bulge_mask_cnt
        if self.peak_bulge_anchor_idx > 0 and peak_idx - self.peak_bulge_anchor_idx > self.af_win_size * 3:
            self.peak_bulge_anchor_idx = -1
            self.peak_bulge_cnt = 0
            logging.info(f'self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        _alarm_arc_th = 2048 / self.indicator_max_val
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _score > _alarm_arc_th else self.alarm_arc_cnt
        # self.alarm_arc_descend_peak_cnt = self.alarm_arc_descend_peak_cnt + 1 if _delta_peak < 0 else 0
        # self.alarm_arc_cnt = self.alarm_arc_cnt - 4 if self.alarm_arc_descend_peak_cnt > 4 else self.alarm_arc_cnt
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _score > _alarm_arc_th else self.alarm_arc_idx_s
        self.alarm_arc_idx_e = peak_idx if self.alarm_arc_idx_s > 0 and _score < 0.1 else self.alarm_arc_idx_e
        if self.alarm_arc_cnt > 4:  # pre-alarm
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 2] * self.af_win_size
            # self.alarm_arc_state = 1
        if self.alarm_arc_idx_s > 0:
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_s - self.af_win_size:self.alarm_arc_idx_s] = \
                [self.indicator_max_val / 4 * 1] * self.af_win_size
        # if self.alarm_arc_idx_e > 0:
        #     self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
        #         [self.indicator_max_val / 4 * 1] * self.af_win_size
        self.last_peak_idx = peak_idx

    def sample(self):
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size, peak_th=self.power_mean + 8)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:  # seq filter
            return
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
        if self.db.db['rt'].seq_state_gt_arc[-1] > 0:
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 2] * self.af_win_size
            self.samples_pos.append(_seq_pick)
        else:
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 4] * self.af_win_size
            self.samples_neg.append(_seq_pick)
