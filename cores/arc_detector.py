# FILE: arc_detector.py

import logging
import os
import random
import sys
from subprocess import *

import numpy as np
import onnx
import pywt
import torch
from onnxsim import simplify
from scipy.signal import *

from cores.classifier import ClassifierCNNAE
from data.data import DataRT
from utils.macros import *


class ArcDetector:
    def __init__(self):
        self._build_model()
        self.ini_peak_cnt = 0
        self.peak_interval_pred = -1
        self.peaks_init = []
        self.power_mean = -1
        self.peak_lr = 1e-3
        self.pm_lr = 1e-4
        self.peak_mean = -1
        self.peak_update_cnt = MEAN_PEAK_UPDATE_CNT_TH
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
        self.peak_eval_win_size = 64  # 64 for 22k sample rate
        self.af_win_size = int(SAMPLE_RATE / 50)
        self.last_peak_idx = -1
        self.last_peak_val = -1
        self.peak_anchor_idx = -1
        self.sample_rate_org = 1 * 1000 * 1000
        self.sample_rate = SAMPLE_RATE  # 256 * 50
        self.sub_sample_rate = int(self.sample_rate_org / self.sample_rate)
        self.sample_rate_new = 22325
        self.filter_cutoff_freq = self.sample_rate_new * 0.4  # hz
        self.filter_order = 4
        # self.filter_b, self.filter_a, self.filter_zi_org = self._design_highpass_filter()
        # self.filter_b, self.filter_a, self.filter_zi_org = self._design_highpass_filter_lp()
        # self.filter_zi = self.filter_zi_org
        self.sample_win_size = self.sample_rate  # 1s
        self.sample_cnt = 1024
        self.samples_neg, self.samples_pos = list(), list()
        self.seq_template = None
        self.seq_template_idx = -1
        self.filter_max, self.filter_th, self.filter_th_cnt = -1, -1, 0
        self.alarm_arc_cnt = 0
        self.alarm_overload_cnt = 0
        self.alarm_lower_cnt = 0
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
        self.sub_sample_cnt = 1
        self.alarm_idle_cnt = 0
        self.feats_ref = []
        self.seq_power_proc_len = 0
        # self.export_onnx()

    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/v9 -  [v8] + data_v9/afdd_models/best_v4.pt'):
    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/v10 - [v9] + data_v8hard/afdd_models - 8gpu/afdd_models_mp_r1/best_e222_b0.8714.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v11 - [v10] + data_v11/afdd_models_mp_r0_e256/best_e185_b0.8828.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v11 - [v10] + data_v11/afdd_models_mp_r1_e512/best_e337_b0.8917.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v11 - [v10] + data_v11/afdd_models_mp_r1_e512/best_e483_b0.9068.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v12 - [v11] + data_v12/afdd_models_mp_r0_e512/best_e474_b0.9193.pt'):
    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/v13 - [v12] + patch_5_6_7_10_11_12/afdd_models_mp_r0/best_e352_b0.9713.pt'):
    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/v14 - [v13] + pos2s/afdd_models_mp_r0/best_e494_b0.9374.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v15 - [v14] + data_v13/afdd_models_mp_r1/best_e506_b0.9398.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/v17 - [v16] + data_v15/afdd_models_mp_r0+/best_e193_b0.9313.pt'):
    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/models_lite/v19 - lite model v0/afdd_models_mp_r0/best_e245_b0.9699.pt'):
    # def _build_model(self, path_model='/home/manu/mnt/ST8000DM004-2U91/afdd/models/models_arm/v7 - [fv6] min-max normal/afdd_models_mp/best_e378_b0.9836.pt'):
    # def _build_model(self, path_model='/home/manu/tmp/afdd_models_mp_v5/best_e323_b0.9766.pt'):
    # def _build_model(self, path_model='/home/manu/tmp/afdd_models_mp_v0/best_e319_b0.9819.pt'):
    # def _build_model(self, path_model='/home/manu/tmp/afdd_models_mp_v1/best_e348_b0.9867.pt'):
    # def _build_model(self, path_model='/home/manu/tmp/afdd_models_mp/best_e472_b0.9878.pt'):
    # def _build_model(self, path_model='/media/manu/ST8000DM004-2U91/afdd/models/models_arm/v9 - dv37/afdd_models_mp/best_e501_b0.9864.pt'):
    def _build_model(self, path_model='/home/manu/tmp/afdd_models_mp/ae_best_e2186_acc0.9756.pt'):
        # with open('/home/manu/tmp/model.pickle', 'rb') as f:
        #     self.classifier = pickle.load(f)
        # self.classifier = ClassifierCNN(args=path_model, is_infer=True)
        import argparse
        classifier_args = argparse.Namespace(
            rank=0,  # 非分布式模式下，rank为0
            path_ckpt=path_model,
            save_dir=None  # 评估时不需要保存目录
        )
        self.classifier = ClassifierCNNAE(args=classifier_args, ddp=False)
        # _state_dict = torch.load(path_model, map_location=torch.device('cuda:0'))
        # _new_state_dict = {}
        # for k, v in _state_dict.items():
        #     if k.startswith('module.'):
        #         _new_state_dict[k[7:]] = v
        #     else:
        #         _new_state_dict[k] = v
        # self.classifier.model.load_state_dict(_new_state_dict)

    # =============================================================
    # 1. 导出并简化 onnx
    # =============================================================
    def export_onnx(self,
                    onnx_path: str = '/home/manu/tmp/classifier_sim.onnx',
                    opset: int = 11,
                    simplify_flag: bool = True):
        self.classifier.model.eval().cpu()

        """
        将当前 self.classifier.model 导出为 onnx，并(可选)做 simplify
        """
        # ①  准备一个 dummy input，形状必须与 infer 时保持一致
        #     这里默认用 arcDetector 推理用到的 self.af_win_size
        # dummy_input = torch.randn(1, self.af_win_size, dtype=torch.float32)
        _channel_in = int(SAMPLE_RATE / 50)
        _channel_in = ((_channel_in // 32) + 1) * 32
        dummy_input = torch.randn(1, 1, _channel_in)

        # ②  导出
        torch.onnx.export(
            self.classifier.model,  # pytorch 模型
            dummy_input,  # 样例输入
            onnx_path,  # 输出文件
            input_names=['input'],
            output_names=['y_pred', 'feats'],
            opset_version=opset,
            # dynamic_axes={
            #     'input': {1: 'seq_len'},  # 允许第二维长度可变
            #     'output': {1: 'seq_len_out'}
            # }
        )
        print(f'[ONNX] export done  ->  {onnx_path}')

        # ③  (可选) simplify
        if simplify_flag:
            print('[ONNX] simplifying ...')
            model = onnx.load(onnx_path)
            model_simp, check = simplify(model)
            assert check, 'simplify failed'
            onnx.save(model_simp, onnx_path)
            print(f'[ONNX] simplify done -> {onnx_path}')

        self.classifier.model.cuda()

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

    def save_seq(self, path_save='/home/manu/tmp/seq_pick.npy'):
        np.save(path_save, (np.array(self.db.db['rt'].seq_power) - 2048) * 40 / 2048)

    def save_samples(self, path_save='/home/manu/tmp/smartsd'):
        # if len(self.samples_neg) > len(self.samples_pos):
        #     self.samples_neg = random.sample(self.samples_neg, len(self.samples_pos))
        logging.info(f'len self.samples_pos -> {len(self.samples_pos)}')
        logging.info(f'len self.samples_neg -> {len(self.samples_neg)}')
        for seq_pick in self.samples_pos:
            self._update_svm_label_file(seq_pick, path_out=path_save, subset='pos')
        for seq_pick in self.samples_neg:
            self._update_svm_label_file(seq_pick, path_out=path_save, subset='neg')

    def reset(self):
        self.last_peak_val = -1
        self.seq_power_proc_len = 0
        self.alarm_idle_cnt = 0
        self.peak_update_cnt = MEAN_PEAK_UPDATE_CNT_TH
        self.ini_peak_cnt = 0
        self.peak_interval_pred = -1
        self.peak_mean = -1
        self.sub_sample_cnt = 1
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
        self.peak_anchor_idx = -1
        self.peak_eval_win.clear()
        self.alarm_arc_cnt = 0
        self.alarm_overload_cnt = 0
        self.alarm_lower_cnt = 0
        self.filter_max, self.filter_th, self.filter_th_cnt = -1, -1, 0
        self.samples_neg.clear()
        self.samples_pos.clear()
        # self.filter_zi = self.filter_zi_org
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

    def _design_highpass_filter_lp(self):
        nyq = 0.5 * self.sample_rate  # TODO
        normal_cutoff = self.filter_cutoff_freq / nyq
        b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)
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

    def _detect_peak(self, cur_val, win_size=256, peak_th=0, reset=False):
        if reset:
            self.peak_eval_win.clear()
            return -1
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

    def _detect_peak_or_valley(self, cur_val, win_size=256, peak_th=0, valley_th=0, reset=False):
        if reset:
            self.peak_eval_win.clear()
            return -1

        self.peak_eval_win.append(cur_val)
        self.peak_eval_win = self.peak_eval_win[-win_size:]

        if len(self.peak_eval_win) < win_size:
            return -1

        peak_candidate_idx = win_size // 2
        peak_candidate_val = self.peak_eval_win[peak_candidate_idx]

        # Check for peak
        if peak_candidate_val == max(self.peak_eval_win) and peak_candidate_val > peak_th:
            self.peak_eval_win.clear()
            return peak_candidate_idx

        # Check for valley
        if peak_candidate_val == min(self.peak_eval_win) and peak_candidate_val < valley_th:
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
        # if self.alarm_overload_cnt > 2:
        #     print('overload alarm !!!')
        #     self.db.db['rt'].seq_state_pred_arc[self.last_peak_idx] = self.indicator_max_val * 64 / 100
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            logging.info(f'self.peak_bulge_mask_cnt --> {self.peak_bulge_mask_cnt}')
            _delta_peak_th = 256 if self.peak_bulge_mask_cnt > 0 else 2048
            _delta_peak = \
                self.db.db['rt'].seq_power[self.alarm_arc_idx_s] - self.db.db['rt'].seq_power[self.alarm_arc_idx_e]
            logging.info(f'_delta_peak --> {_delta_peak}')
            # self.alarm_arc_cnt = self.alarm_arc_cnt - 16 if _delta_peak > _delta_peak_th else self.alarm_arc_cnt
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
            if _hf_cnt > 0 and self.alarm_arc_cnt > 2:
                logging.info(f'alarm idx --> {self.last_peak_idx}')
                self.db.db['rt'].seq_state_pred_arc[self.last_peak_idx - self.af_win_size:self.last_peak_idx] = \
                    [self.indicator_max_val * 99 / 100] * self.af_win_size
                self.alarm_arc_state = 0
                print('arc fault alarm !!!')
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1
        power_pick = self.db.db['rt'].seq_power[-1]
        # logging.info(f'power_pick --> {power_pick}')
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size, peak_th=self.power_mean + 128)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 2:  # seq filter
            self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0000001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.peak_bulge_mask_cnt = \
                self.peak_bulge_mask_cnt - 1 if self.peak_bulge_mask_cnt > 0 else self.peak_bulge_mask_cnt
            # self.alarm_arc_cnt = max(0, self.alarm_arc_cnt)
            self.peak_miss_cnt += 1
            return
        self.peak_miss_cnt = 0
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        self.alarm_overload_cnt = \
            self.alarm_overload_cnt + 1 if _peak_val > 5000 else self.alarm_overload_cnt
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _th_arc = (_peak_val - self.power_mean) * 0.01
        _cnt_arc = np.sum(np.abs(_seq_pick_power - self.power_mean) < _th_arc)
        _scale_arc = 128
        _cnt_arc = _cnt_arc if _cnt_arc * _scale_arc < 4096 else 0
        self.db.db['rt'].seq_state_pred_balcony[peak_idx - self.af_win_size:peak_idx] = \
            [_cnt_arc * _scale_arc] * self.af_win_size
        _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score = self.classifier.infer(_data, batch_size=1)[0]
        # logging.info(f'_score --> {_score}')
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
        _th_raw = 2048
        _alarm_arc_th = _th_raw / self.indicator_max_val
        # _is_arc = _score > _alarm_arc_th or _cnt_arc * _scale_arc > _th_raw
        _model_w = 1.0
        _is_arc = (_model_w * _score * self.indicator_max_val + (1 - _model_w) * _cnt_arc * _scale_arc) > _th_raw
        # self.alarm_arc_cnt = \
        #     self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        self.alarm_arc_cnt += _score
        # self.alarm_arc_descend_peak_cnt = self.alarm_arc_descend_peak_cnt + 1 if _delta_peak < 0 else 0
        # self.alarm_arc_cnt = self.alarm_arc_cnt - 4 if self.alarm_arc_descend_peak_cnt > 4 else self.alarm_arc_cnt
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _is_arc else self.alarm_arc_idx_s
        self.alarm_arc_idx_e = peak_idx if self.alarm_arc_idx_s > 0 and not _is_arc else self.alarm_arc_idx_e
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

    @staticmethod
    def _cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _max_cosine_similarity(self, _feat):
        max_similarity = -1
        best_match = None
        for _feat_ref in self.feats_ref:
            similarity = self._cosine_similarity(_feat, _feat_ref)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = _feat_ref
        return max_similarity, best_match

    def infer_v3(self, feat_sample=False):
        _alarm_arc_cnt_th = 4
        _min_val_th = self.indicator_max_val / 2 * 0.05  # MIN_VAL_TH * 2
        _th_raw = 2048 * 1.0
        _ini_peak_cnt_th = 64
        # _is_alarm_overload = self.alarm_overload_cnt > 0 and self.ini_peak_cnt > _ini_peak_cnt_th
        _is_alarm_overload = False
        if self.alarm_idle_cnt > 0:
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val
        if self.ini_peak_cnt < _ini_peak_cnt_th:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 4
        if _is_alarm_overload:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val
        if self.peak_miss_cnt > self.af_win_size * 16:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 8
            self.ini_peak_cnt = 0
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        # assign self.alarm_arc_idx_e when long time peak miss
        if self.peak_miss_cnt > self.af_win_size * 4 and self.alarm_arc_idx_e < 0 < self.alarm_arc_idx_s:
            self.alarm_arc_idx_e = self.db.db['rt'].seq_len
            # self.db.db['rt'].info_pred_peaks.append(self.db.db['rt'].seq_len)
            # self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        # if self.alarm_overload_cnt > 2:
        #     print('overload alarm !!!')
        #     self.db.db['rt'].seq_state_pred_arc[self.last_peak_idx] = self.indicator_max_val * 64 / 100
        _alarm_indicate_scale = ALARM_INDICATE_SCALE
        if self.alarm_idle_cnt > 0 and self.ini_peak_cnt == 0:  # idle alarm confirm
            print('arc fault alarm v0 !!!')
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val * _alarm_indicate_scale
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1] = self.indicator_max_val * 1.5  # affect demo_pico !
            self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 1)
            self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                [self.indicator_max_val / 8 * 1] * self.af_win_size
            # logging.info(f'self.peak_bulge_mask_cnt --> {self.peak_bulge_mask_cnt}')
            _delta_peak_th = 256 if self.peak_bulge_mask_cnt > 0 else 2048
            _delta_peak = \
                self.db.db['rt'].seq_power[self.alarm_arc_idx_s] - self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1]
            # logging.info(f'_delta_peak --> {_delta_peak}')
            # self.alarm_arc_cnt = self.alarm_arc_cnt - 16 if _delta_peak > _delta_peak_th else self.alarm_arc_cnt
            _alarm_arc_end_idx = self.last_peak_idx
            _idx_e = self.alarm_arc_idx_e + self.af_win_size * 4
            _idx_e = _idx_e if _idx_e < self.db.db['rt'].seq_len else self.db.db['rt'].seq_len
            _seq_pick_hf_alarm = \
                np.array(
                    self.db.db['rt'].seq_hf[self.alarm_arc_idx_s - self.af_win_size:_idx_e]).astype(float)
            _hf_cnt = np.sum(_seq_pick_hf_alarm > 0)
            # logging.info(f'_delta_peak_th --> {_delta_peak_th}')
            # logging.info(f'_hf_cnt --> {_hf_cnt}')
            # logging.info(f'self.alarm_arc_cnt --> {self.alarm_arc_cnt}')
            # logging.info(f'self.alarm_arc_idx_s --> {self.alarm_arc_idx_s}')
            # logging.info(f'self.last_peak_idx --> {self.last_peak_idx}')
            if _hf_cnt > 0 and self.alarm_arc_cnt > _alarm_arc_cnt_th:
                # logging.info(f'alarm idx --> {self.last_peak_idx}')
                # self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                #     [self.indicator_max_val * 99 / 100] * self.af_win_size
                self.alarm_arc_state = 0
                if self.ini_peak_cnt < _ini_peak_cnt_th:  # idle state alarm
                    self.alarm_idle_cnt = 8
                    # logging.info(f'self.peak_miss_cnt --> {self.peak_miss_cnt}')
                    # if self.peak_miss_cnt > self.af_win_size * 1.5:
                    #     print('arc fault alarm !!!')
                    #     self.db.db['rt'].seq_state_pred_arc[
                    #     self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                    #         [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
                else:
                    print('arc fault alarm v1 !!!')
                    self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                        [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1
        power_pick = self.db.db['rt'].seq_power[-1]
        _lower_th, _lower_cnt_set = 512, 1
        self.alarm_lower_cnt = _lower_cnt_set if power_pick < _lower_th else self.alarm_lower_cnt
        # logging.info(f'power_pick --> {power_pick}')
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        _scale_peak_mean = 1.
        self.db.db['rt'].seq_peak_mean[-1] = self.peak_mean * _scale_peak_mean if self.peak_mean > 0 else 0
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + _min_val_th)
        # peak_idx_norm = self._detect_peak_or_valley(power_pick, win_size=self.peak_eval_win_size,
        #                                             peak_th=self.power_mean + _min_val_th,
        #                                             valley_th=self.power_mean - _min_val_th)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        _peak_suppose_idx = int(self.peak_anchor_idx + self.peak_interval_pred) if self.peak_interval_pred > 0 else -1
        if _peak_suppose_idx > 0 and _peak_suppose_idx < self.db.db['rt'].seq_len:
            self.db.db['rt'].info_eval_peaks.append(_peak_suppose_idx)
            self.peak_anchor_idx = _peak_suppose_idx
        if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 1.5:  # seq filter
            self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            self.alarm_lower_cnt = self.alarm_lower_cnt - 0.0001 if self.alarm_lower_cnt > 0 else self.alarm_lower_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.alarm_idle_cnt = self.alarm_idle_cnt - 0.0001 if self.alarm_idle_cnt > 0 else self.alarm_idle_cnt
            self.peak_bulge_mask_cnt = \
                self.peak_bulge_mask_cnt - 1 if self.peak_bulge_mask_cnt > 0 else self.peak_bulge_mask_cnt
            # self.alarm_arc_cnt = max(0, self.alarm_arc_cnt)
            self.peak_miss_cnt += 1
            return
        self.ini_peak_cnt += 1
        _peak_interval_current = peak_idx - self.last_peak_idx if self.last_peak_idx > 0 else -1
        self.peak_interval_pred = \
            self.peak_interval_pred * (
                    1 - self.pm_lr) + _peak_interval_current * self.pm_lr if self.peak_interval_pred > 0 \
                else _peak_interval_current
        self.peak_anchor_idx = peak_idx
        self.peak_miss_cnt = 0
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        self.peak_update_cnt = self.peak_update_cnt + 1 if abs(self.peak_mean - _peak_val) > 64 else 0
        self.peak_mean = self.peak_mean * (
                1 - self.peak_lr) + _peak_val * self.peak_lr if self.peak_update_cnt < MEAN_PEAK_UPDATE_CNT_TH else _peak_val
        # self.alarm_overload_cnt = self.alarm_overload_cnt + 1 if _peak_val > self.indicator_max_val else self.alarm_overload_cnt
        _alarm_overload_cnt_set_val = 1
        self.alarm_overload_cnt = _alarm_overload_cnt_set_val if _peak_val > self.indicator_max_val else self.alarm_overload_cnt
        # if peak_idx - self.last_peak_idx > self.af_win_size * 1.2 and len(self.db.db['rt'].info_eval_peaks) > 1:
        #     _peak_idx_pad = \
        #         self.db.db['rt'].info_eval_peaks[-1] if abs(
        #             peak_idx - self.db.db['rt'].info_eval_peaks[-1]) > self.af_win_size / 8 \
        #             else self.db.db['rt'].info_eval_peaks[-2]
        #     self.db.db['rt'].info_pred_peaks.append(_peak_idx_pad)
        #     _seq_pick_power_pad = np.array(
        #         self.db.db['rt'].seq_power[_peak_idx_pad - self.af_win_size:_peak_idx_pad]).astype(float)
        #     _seq_pick_hf_pad = np.array(self.db.db['rt'].seq_hf[_peak_idx_pad - self.af_win_size:_peak_idx_pad]).astype(
        #         float)
        #     _seq_pick_pad = np.concatenate((_seq_pick_power_pad, _seq_pick_hf_pad), axis=0)
        #     _data_pad = _seq_pick_pad[np.newaxis, :]
        #     _score_pad = self.classifier.infer(_data_pad, batch_size=1)[0]
        #     self.db.db['rt'].seq_state_pred_classifier[_peak_idx_pad - self.af_win_size:_peak_idx_pad] = \
        #         [_score_pad * self.indicator_max_val] * self.af_win_size
        #     self.db.db['rt'].info_af_scores.append(_score_pad)
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _th_arc = (_peak_val - self.power_mean) * 0.01
        _cnt_arc = np.sum(np.abs(_seq_pick_power - self.power_mean) < _th_arc)
        _scale_arc = 128
        _cnt_arc = _cnt_arc if _cnt_arc * _scale_arc < 4096 else 4096 / _scale_arc
        self.db.db['rt'].seq_state_pred_balcony[peak_idx - self.af_win_size:peak_idx] = \
            [_cnt_arc * _scale_arc] * self.af_win_size
        _th_peak_duty = _peak_val * 0.5
        _duty_cycle = np.mean(_seq_pick_power > _th_peak_duty)
        # logging.info(f'_duty_cycle --> {_duty_cycle}')
        # _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
        # _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
        _seq_pick = _seq_pick_power
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score, _feat = self.classifier.infer(_data, batch_size=1)
        _score = _score[0]
        # logging.info(f'_score --> {_score}')
        # logging.info(f'_feat --> {_feat}')
        if not feat_sample:
            _max_similarity, _ = self._max_cosine_similarity(_feat)
            # logging.info(f'_max_similarity --> {_max_similarity}')
            _score = 0 if _max_similarity > 0.999 else _score
        self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
            [_score * self.indicator_max_val] * self.af_win_size
        _delta_peak = self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.last_peak_idx]
        _bulge_th = 512
        if _delta_peak > _bulge_th and self.peak_bulge_anchor_idx < 0 and self.last_peak_idx > 0:
            self.peak_bulge_anchor_idx = self.last_peak_idx
            self.peak_bulge_cnt += 1
            # logging.info(f'_delta_peak -- > {_delta_peak}')
            # logging.info(f'int self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        if self.peak_bulge_anchor_idx > 0:
            _delta_peak_bulge = \
                self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.peak_bulge_anchor_idx]
            self.peak_bulge_cnt = self.peak_bulge_cnt + 1 if _delta_peak_bulge > _bulge_th else self.peak_bulge_cnt
            # logging.info(f'self.peak_bulge_cnt --> {self.peak_bulge_cnt}')
        self.peak_bulge_mask_cnt = SAMPLE_RATE if self.peak_bulge_cnt > 2 else self.peak_bulge_mask_cnt
        if self.peak_bulge_anchor_idx > 0 and peak_idx - self.peak_bulge_anchor_idx > self.af_win_size * 3:
            self.peak_bulge_anchor_idx = -1
            self.peak_bulge_cnt = 0
            # logging.info(f'self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        # _th_raw = _th_raw / 2 if _peak_val > self.indicator_max_val else _th_raw
        _alarm_arc_th = _th_raw / self.indicator_max_val
        # _is_arc = _score > _alarm_arc_th or _cnt_arc * _scale_arc > _th_raw
        _model_w = 1.0
        _is_arc = (_model_w * _score * self.indicator_max_val + (1 - _model_w) * _cnt_arc * _scale_arc) > _th_raw
        if _is_arc and feat_sample:
            self.feats_ref.append(_feat)
            logging.info(f'len(self.feats_ref) --> {len(self.feats_ref)}')

        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        # self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc and _is_alarm_overload else self.alarm_arc_cnt
        if (self.af_win_size * 1.5 < peak_idx - self.last_peak_idx < self.af_win_size * 8
                and self.alarm_arc_cnt > 0.5 and _peak_val > self.indicator_max_val):
            self.alarm_arc_cnt += (peak_idx - self.last_peak_idx) / self.af_win_size
            # logging.info(f'padding [{peak_idx}] self.alarm_arc_cnt -- > {self.alarm_arc_cnt}')
            self.db.db['rt'].seq_state_gt_normal[self.last_peak_idx:peak_idx] = \
                [self.indicator_max_val / 8] * (peak_idx - self.last_peak_idx)

        if _duty_cycle < 0.1 and _peak_val > self.indicator_max_val:
            self.alarm_arc_cnt += 1
            self.db.db['rt'].seq_state_gt_normal[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 8] * self.af_win_size

        # self.alarm_arc_cnt += _score
        # self.alarm_arc_descend_peak_cnt = self.alarm_arc_descend_peak_cnt + 1 if _delta_peak < 0 else 0
        # self.alarm_arc_cnt = self.alarm_arc_cnt - 4 if self.alarm_arc_descend_peak_cnt > 4 else self.alarm_arc_cnt
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _is_arc and self.alarm_idle_cnt <= 0 \
            else self.alarm_arc_idx_s
        # if self.alarm_arc_idx_s > 0 and not _is_arc and self.ini_peak_cnt > _ini_peak_cnt_th:
        if self.alarm_arc_idx_s > 0 and not _is_arc and self.alarm_idle_cnt <= 0:
            # _drop = self.peak_mean - self.db.db['rt'].seq_power[peak_idx]
            # _drop_th = (self.peak_mean - self.power_mean) / 2.5
            _drop, _drop_th = 1, -1  # disable
            if _drop > _drop_th:
                # logging.info(f'dropping --> {_drop} [{_drop_th}]')
                self.alarm_arc_idx_e = peak_idx
                # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 2] = self.indicator_max_val * 1.8
                # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 2)
                # self.db.db['rt'].info_af_scores.append(_drop)
                # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 3] = self.indicator_max_val * 1.7
                # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 3)
                # self.db.db['rt'].info_af_scores.append(_drop_th)
        if (self.alarm_arc_cnt > _alarm_arc_cnt_th
                and self.db.db['rt'].seq_state_pred_arc[peak_idx - 1] < self.indicator_max_val):  # pre-alarm
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 4 * 3] * self.af_win_size
            # self.alarm_arc_state = 1
        if self.alarm_arc_idx_s > 0:
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_s - self.af_win_size:self.alarm_arc_idx_s] = \
                [self.indicator_max_val / 4 * 1] * self.af_win_size
        # if self.alarm_arc_idx_e > 0:
        #     self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
        #         [self.indicator_max_val / 4 * 1] * self.af_win_size
        # _seq_pick_power_last = np.ones(len(_seq_pick_power)) * self.indicator_max_val / 2.
        # if self.last_peak_idx > 0:
        #     _seq_pick_power_last = np.array(
        #         self.db.db['rt'].seq_power[self.last_peak_idx - self.af_win_size:self.last_peak_idx]).astype(float)
        # _integral1, _integral2 = np.sum(_seq_pick_power), np.sum(_seq_pick_power_last)
        # _integral_change_ratio = (_integral2 - _integral1) / _integral1
        # self.db.db['rt'].seq_state_pred_icr[peak_idx - self.af_win_size:peak_idx] = \
        #     [_integral_change_ratio * self.indicator_max_val * 64] * self.af_win_size
        self.last_peak_idx = peak_idx

    def infer_v4(self, feat_sample=False):
        if self.db.db['rt'].seq_len % (self.af_win_size * 3) >= (self.af_win_size * 2):  # data missing simulation
            self.peak_eval_win.clear()  # !
            self.seq_power_proc_len = 0  # !
            return
        self.seq_power_proc_len += 1
        # logging.info(f"self.seq_power_proc_len --> {self.seq_power_proc_len}")
        if self.seq_power_proc_len < self.af_win_size:  # !
            return
        # logging.info(f"self.seq_power_proc_len --> {self.seq_power_proc_len}")
        _alarm_arc_cnt_th = 1.5
        _min_val_th = self.indicator_max_val / 2 * 0.05  # MIN_VAL_TH * 2
        _th_raw = 2048 * 1.0
        _ini_peak_cnt_th = 64
        # _is_alarm_overload = self.alarm_overload_cnt > 0 and self.ini_peak_cnt > _ini_peak_cnt_th
        # _is_alarm_overload = False
        if self.alarm_idle_cnt > 0:
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val
        if self.ini_peak_cnt < _ini_peak_cnt_th:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 4
        # if _is_alarm_overload:
        #     self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val
        if self.peak_miss_cnt > self.af_win_size * 16:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 8
            self.ini_peak_cnt = 0
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        # assign self.alarm_arc_idx_e when long time peak miss
        if self.peak_miss_cnt > self.af_win_size * 4 and self.alarm_arc_idx_e < 0 < self.alarm_arc_idx_s:
            self.alarm_arc_idx_e = self.db.db['rt'].seq_len
            # self.db.db['rt'].info_pred_peaks.append(self.db.db['rt'].seq_len)
            # self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        # if self.alarm_overload_cnt > 2:
        #     print('overload alarm !!!')
        #     self.db.db['rt'].seq_state_pred_arc[self.last_peak_idx] = self.indicator_max_val * 64 / 100
        _alarm_indicate_scale = ALARM_INDICATE_SCALE
        if self.alarm_idle_cnt > 0 and self.ini_peak_cnt == 0:  # idle alarm confirm
            print('arc fault alarm v0 !!!')
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val * _alarm_indicate_scale
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1] = self.indicator_max_val * 1.5  # affect demo_pico !
            self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 1)
            self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                [self.indicator_max_val / 8 * 1] * self.af_win_size
            # logging.info(f'self.peak_bulge_mask_cnt --> {self.peak_bulge_mask_cnt}')
            # _delta_peak_th = 256 if self.peak_bulge_mask_cnt > 0 else 2048
            # _delta_peak = \
            #     self.db.db['rt'].seq_power[self.alarm_arc_idx_s] - self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1]
            # logging.info(f'_delta_peak --> {_delta_peak}')
            # self.alarm_arc_cnt = self.alarm_arc_cnt - 16 if _delta_peak > _delta_peak_th else self.alarm_arc_cnt
            # _alarm_arc_end_idx = self.last_peak_idx
            # _idx_e = self.alarm_arc_idx_e + self.af_win_size * 4
            # _idx_e = _idx_e if _idx_e < self.db.db['rt'].seq_len else self.db.db['rt'].seq_len
            # _seq_pick_hf_alarm = \
            #     np.array(
            #         self.db.db['rt'].seq_hf[self.alarm_arc_idx_s - self.af_win_size:_idx_e]).astype(float)
            # _hf_cnt = np.sum(_seq_pick_hf_alarm > 0)
            # logging.info(f'_delta_peak_th --> {_delta_peak_th}')
            # logging.info(f'_hf_cnt --> {_hf_cnt}')
            # logging.info(f'self.alarm_arc_cnt --> {self.alarm_arc_cnt}')
            # logging.info(f'self.alarm_arc_idx_s --> {self.alarm_arc_idx_s}')
            # logging.info(f'self.last_peak_idx --> {self.last_peak_idx}')
            if self.alarm_arc_cnt > _alarm_arc_cnt_th:
                # logging.info(f'alarm idx --> {self.last_peak_idx}')
                # self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                #     [self.indicator_max_val * 99 / 100] * self.af_win_size
                self.alarm_arc_state = 0
                if self.ini_peak_cnt < _ini_peak_cnt_th:  # idle state alarm
                    self.alarm_idle_cnt = 8
                    # logging.info(f'self.peak_miss_cnt --> {self.peak_miss_cnt}')
                    # if self.peak_miss_cnt > self.af_win_size * 1.5:
                    #     print('arc fault alarm !!!')
                    #     self.db.db['rt'].seq_state_pred_arc[
                    #     self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                    #         [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
                else:
                    print('arc fault alarm v1 !!!')
                    self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                        [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1

        power_pick = self.db.db['rt'].seq_power[-1]
        # _lower_th, _lower_cnt_set = 512, 1
        # self.alarm_lower_cnt = _lower_cnt_set if power_pick < _lower_th else self.alarm_lower_cnt
        # logging.info(f'power_pick --> {power_pick}')
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        # logging.info(f'self.power_mean --> {self.power_mean}')
        # _scale_peak_mean = 1.
        # self.db.db['rt'].seq_peak_mean[-1] = self.peak_mean * _scale_peak_mean if self.peak_mean > 0 else 0
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + _min_val_th)
        # peak_idx_norm = self._detect_peak_or_valley(power_pick, win_size=self.peak_eval_win_size,
        #                                             peak_th=self.power_mean + _min_val_th,
        #                                             valley_th=self.power_mean - _min_val_th)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        # _peak_suppose_idx = int(self.peak_anchor_idx + self.peak_interval_pred) if self.peak_interval_pred > 0 else -1
        # if _peak_suppose_idx > 0 and _peak_suppose_idx < self.db.db['rt'].seq_len:
        #     self.db.db['rt'].info_eval_peaks.append(_peak_suppose_idx)
        #     self.peak_anchor_idx = _peak_suppose_idx
        if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 1.5:  # seq filter
            # self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            # self.alarm_lower_cnt = self.alarm_lower_cnt - 0.0001 if self.alarm_lower_cnt > 0 else self.alarm_lower_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.alarm_idle_cnt = self.alarm_idle_cnt - 0.0001 if self.alarm_idle_cnt > 0 else self.alarm_idle_cnt
            # self.peak_bulge_mask_cnt = \
            #     self.peak_bulge_mask_cnt - 1 if self.peak_bulge_mask_cnt > 0 else self.peak_bulge_mask_cnt
            # self.alarm_arc_cnt = max(0, self.alarm_arc_cnt)
            self.peak_miss_cnt += 1
            return
        self.ini_peak_cnt += 1
        # _peak_interval_current = peak_idx - self.last_peak_idx if self.last_peak_idx > 0 else -1
        # self.peak_interval_pred = \
        #     self.peak_interval_pred * (
        #             1 - self.pm_lr) + _peak_interval_current * self.pm_lr if self.peak_interval_pred > 0 \
        #         else _peak_interval_current
        self.peak_anchor_idx = peak_idx
        self.peak_miss_cnt = 0
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        # self.peak_update_cnt = self.peak_update_cnt + 1 if abs(self.peak_mean - _peak_val) > 64 else 0
        # self.peak_mean = self.peak_mean * (
        #         1 - self.peak_lr) + _peak_val * self.peak_lr if self.peak_update_cnt < MEAN_PEAK_UPDATE_CNT_TH else _peak_val
        # self.alarm_overload_cnt = self.alarm_overload_cnt + 1 if _peak_val > self.indicator_max_val else self.alarm_overload_cnt
        # _alarm_overload_cnt_set_val = 1
        # self.alarm_overload_cnt = _alarm_overload_cnt_set_val if _peak_val > self.indicator_max_val else self.alarm_overload_cnt
        # if peak_idx - self.last_peak_idx > self.af_win_size * 1.2 and len(self.db.db['rt'].info_eval_peaks) > 1:
        #     _peak_idx_pad = \
        #         self.db.db['rt'].info_eval_peaks[-1] if abs(
        #             peak_idx - self.db.db['rt'].info_eval_peaks[-1]) > self.af_win_size / 8 \
        #             else self.db.db['rt'].info_eval_peaks[-2]
        #     self.db.db['rt'].info_pred_peaks.append(_peak_idx_pad)
        #     _seq_pick_power_pad = np.array(
        #         self.db.db['rt'].seq_power[_peak_idx_pad - self.af_win_size:_peak_idx_pad]).astype(float)
        #     _seq_pick_hf_pad = np.array(self.db.db['rt'].seq_hf[_peak_idx_pad - self.af_win_size:_peak_idx_pad]).astype(
        #         float)
        #     _seq_pick_pad = np.concatenate((_seq_pick_power_pad, _seq_pick_hf_pad), axis=0)
        #     _data_pad = _seq_pick_pad[np.newaxis, :]
        #     _score_pad = self.classifier.infer(_data_pad, batch_size=1)[0]
        #     self.db.db['rt'].seq_state_pred_classifier[_peak_idx_pad - self.af_win_size:_peak_idx_pad] = \
        #         [_score_pad * self.indicator_max_val] * self.af_win_size
        #     self.db.db['rt'].info_af_scores.append(_score_pad)
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        # _th_arc = (_peak_val - self.power_mean) * 0.01
        # _cnt_arc = np.sum(np.abs(_seq_pick_power - self.power_mean) < _th_arc)
        # _scale_arc = 128
        # _cnt_arc = _cnt_arc if _cnt_arc * _scale_arc < 4096 else 4096 / _scale_arc
        # self.db.db['rt'].seq_state_pred_balcony[peak_idx - self.af_win_size:peak_idx] = \
        #     [_cnt_arc * _scale_arc] * self.af_win_size
        # _th_peak_duty = _peak_val * 0.5
        # _duty_cycle = np.mean(_seq_pick_power > _th_peak_duty)
        # logging.info(f'_duty_cycle --> {_duty_cycle}')
        # _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
        # _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
        _seq_pick = _seq_pick_power
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score, _feat = self.classifier.infer(_data, batch_size=1)
        _score = _score[0]
        # logging.info(f'_score --> {_score}')
        # logging.info(f'_feat --> {_feat}')
        # if not feat_sample:
        #     _max_similarity, _ = self._max_cosine_similarity(_feat)
        #     # logging.info(f'_max_similarity --> {_max_similarity}')
        #     _score = 0 if _max_similarity > 0.999 else _score
        self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
            [_score * self.indicator_max_val] * self.af_win_size
        # _delta_peak = self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.last_peak_idx]
        # _bulge_th = 512
        # if _delta_peak > _bulge_th and self.peak_bulge_anchor_idx < 0 and self.last_peak_idx > 0:
        #     self.peak_bulge_anchor_idx = self.last_peak_idx
        #     self.peak_bulge_cnt += 1
        #     # logging.info(f'_delta_peak -- > {_delta_peak}')
        #     # logging.info(f'int self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        # if self.peak_bulge_anchor_idx > 0:
        #     _delta_peak_bulge = \
        #         self.db.db['rt'].seq_power[peak_idx] - self.db.db['rt'].seq_power[self.peak_bulge_anchor_idx]
        #     self.peak_bulge_cnt = self.peak_bulge_cnt + 1 if _delta_peak_bulge > _bulge_th else self.peak_bulge_cnt
        #     # logging.info(f'self.peak_bulge_cnt --> {self.peak_bulge_cnt}')
        # self.peak_bulge_mask_cnt = SAMPLE_RATE if self.peak_bulge_cnt > 2 else self.peak_bulge_mask_cnt
        # if self.peak_bulge_anchor_idx > 0 and peak_idx - self.peak_bulge_anchor_idx > self.af_win_size * 3:
        #     self.peak_bulge_anchor_idx = -1
        #     self.peak_bulge_cnt = 0
        # logging.info(f'self.peak_bulge_anchor_idx --> {self.peak_bulge_anchor_idx}')
        # _th_raw = _th_raw / 2 if _peak_val > self.indicator_max_val else _th_raw
        _alarm_arc_th = _th_raw / self.indicator_max_val
        # _is_arc = _score > _alarm_arc_th or _cnt_arc * _scale_arc > _th_raw
        # _model_w = 1.0
        _is_arc = _score * self.indicator_max_val > _th_raw
        # if _is_arc and feat_sample:
        #     self.feats_ref.append(_feat)
        #     logging.info(f'len(self.feats_ref) --> {len(self.feats_ref)}')

        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        # self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc and _is_alarm_overload else self.alarm_arc_cnt
        # if (self.af_win_size * 1.5 < peak_idx - self.last_peak_idx < self.af_win_size * 8
        #         and self.alarm_arc_cnt > 0.5 and _peak_val > self.indicator_max_val):
        #     self.alarm_arc_cnt += (peak_idx - self.last_peak_idx) / self.af_win_size
        #     # logging.info(f'padding [{peak_idx}] self.alarm_arc_cnt -- > {self.alarm_arc_cnt}')
        #     self.db.db['rt'].seq_state_gt_normal[self.last_peak_idx:peak_idx] = \
        #         [self.indicator_max_val / 8] * (peak_idx - self.last_peak_idx)

        # if _duty_cycle < 0.1 and _peak_val > self.indicator_max_val:
        #     self.alarm_arc_cnt += 1
        #     self.db.db['rt'].seq_state_gt_normal[peak_idx - self.af_win_size:peak_idx] = \
        #         [self.indicator_max_val / 8] * self.af_win_size

        # self.alarm_arc_cnt += _score
        # self.alarm_arc_descend_peak_cnt = self.alarm_arc_descend_peak_cnt + 1 if _delta_peak < 0 else 0
        # self.alarm_arc_cnt = self.alarm_arc_cnt - 4 if self.alarm_arc_descend_peak_cnt > 4 else self.alarm_arc_cnt
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _is_arc and self.alarm_idle_cnt <= 0 \
            else self.alarm_arc_idx_s
        # if self.alarm_arc_idx_s > 0 and not _is_arc and self.ini_peak_cnt > _ini_peak_cnt_th:
        if self.alarm_arc_idx_s > 0 and not _is_arc and self.alarm_idle_cnt <= 0:
            # _drop = self.peak_mean - self.db.db['rt'].seq_power[peak_idx]
            # _drop_th = (self.peak_mean - self.power_mean) / 2.5
            _drop, _drop_th = 1, -1  # disable
            if _drop > _drop_th:
                # logging.info(f'dropping --> {_drop} [{_drop_th}]')
                self.alarm_arc_idx_e = peak_idx
                # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 2] = self.indicator_max_val * 1.8
                # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 2)
                # self.db.db['rt'].info_af_scores.append(_drop)
                # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 3] = self.indicator_max_val * 1.7
                # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 3)
                # self.db.db['rt'].info_af_scores.append(_drop_th)
        if (self.alarm_arc_cnt > _alarm_arc_cnt_th
                and self.db.db['rt'].seq_state_pred_arc[peak_idx - 1] < self.indicator_max_val):  # pre-alarm
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 4 * 3] * self.af_win_size
            # self.alarm_arc_state = 1
            print('arc fault alarm v2 !!!')  # !
        if self.alarm_arc_idx_s > 0:
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_s - self.af_win_size:self.alarm_arc_idx_s] = \
                [self.indicator_max_val / 4 * 1] * self.af_win_size
        # if self.alarm_arc_idx_e > 0:
        #     self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
        #         [self.indicator_max_val / 4 * 1] * self.af_win_size
        # _seq_pick_power_last = np.ones(len(_seq_pick_power)) * self.indicator_max_val / 2.
        # if self.last_peak_idx > 0:
        #     _seq_pick_power_last = np.array(
        #         self.db.db['rt'].seq_power[self.last_peak_idx - self.af_win_size:self.last_peak_idx]).astype(float)
        # _integral1, _integral2 = np.sum(_seq_pick_power), np.sum(_seq_pick_power_last)
        # _integral_change_ratio = (_integral2 - _integral1) / _integral1
        # self.db.db['rt'].seq_state_pred_icr[peak_idx - self.af_win_size:peak_idx] = \
        #     [_integral_change_ratio * self.indicator_max_val * 64] * self.af_win_size
        self.last_peak_idx = peak_idx

    def infer_v6(self, feat_sample=False):
        self.seq_power_proc_len += 1
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        _alarm_arc_cnt_th = 1.5
        _min_val_th = self.indicator_max_val / 2 * 0.05  # MIN_VAL_TH * 2
        _th_raw = 2048 * 1.0
        _ini_peak_cnt_th = 64
        if self.alarm_idle_cnt > 0:
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val
        if self.ini_peak_cnt < _ini_peak_cnt_th:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 4
        if self.peak_miss_cnt > self.af_win_size * 8:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 8
            self.ini_peak_cnt = 0
        if self.seq_power_proc_len < self.af_win_size:  # !
            return
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        if self.peak_miss_cnt > self.af_win_size * 4 and self.alarm_arc_idx_e < 0 < self.alarm_arc_idx_s:
            self.alarm_arc_idx_e = self.db.db['rt'].seq_len
            logging.info(f'end v1 -- > {self.alarm_arc_cnt}')
        _alarm_indicate_scale = ALARM_INDICATE_SCALE
        if self.alarm_idle_cnt > 0 and self.ini_peak_cnt == 0:  # idle alarm confirm
            print('arc fault alarm v0 !!!')
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val * _alarm_indicate_scale
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1] = self.indicator_max_val * 1.5  # affect demo_pico !
            # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 1)
            # self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                [self.indicator_max_val / 8 * 1] * self.af_win_size
            if self.alarm_arc_cnt > _alarm_arc_cnt_th:
                self.alarm_arc_state = 0
                if self.ini_peak_cnt < _ini_peak_cnt_th:  # idle state alarm
                    self.alarm_idle_cnt = 8
                else:
                    print('arc fault alarm v1 !!!')
                    self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                        [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1

        # power_pick = self.db.db['rt'].seq_power[-1]
        # if self.db.db['rt'].seq_len == 458:
        #     print("manu")
        # self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
        #     else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        # self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + _min_val_th)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        # if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 1.5:  # seq filter
        if peak_idx < 0:  # seq filter
            # self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            # self.alarm_lower_cnt = self.alarm_lower_cnt - 0.0001 if self.alarm_lower_cnt > 0 else self.alarm_lower_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.alarm_idle_cnt = self.alarm_idle_cnt - 0.0005 if self.alarm_idle_cnt > 0 else self.alarm_idle_cnt
            self.peak_miss_cnt += 1
            return
        self.ini_peak_cnt += 1
        self.peak_anchor_idx = peak_idx
        self.peak_miss_cnt = 0
        if self.ini_peak_cnt < 2:  # !
            return
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick = _seq_pick_power
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score = self.classifier.infer(_data, batch_size=1)
        _score = _score[0] * 1e5
        # if _score * self.indicator_max_val > 30:
        #     print("manu")
        self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
            [_score * self.indicator_max_val] * self.af_win_size
        _alarm_arc_th = _th_raw / self.indicator_max_val
        _is_arc = _score * self.indicator_max_val > _th_raw
        # self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt - 0.5 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt  # !
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc and _peak_val > self.indicator_max_val * 0.9 else self.alarm_arc_cnt  # !
        # if _is_arc and _peak_val > self.indicator_max_val * 0.9:
        #     print("2")
        # elif _is_arc:
        #     print("1")
        if self.alarm_arc_cnt > 0:
            logging.info(f'self.alarm_arc_cnt -- > {self.alarm_arc_cnt}')
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _is_arc and self.alarm_idle_cnt <= 0 \
            else self.alarm_arc_idx_s
        if self.alarm_arc_idx_s > 0 and not _is_arc and self.alarm_idle_cnt <= 0:
            self.alarm_arc_idx_e = peak_idx
            logging.info(f'end v0 -- > {self.alarm_arc_cnt}')
        if (self.alarm_arc_cnt > _alarm_arc_cnt_th
                and self.db.db['rt'].seq_state_pred_arc[peak_idx - 1] < self.indicator_max_val):  # pre-alarm
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 4 * 3] * self.af_win_size
            # print('arc fault alarm v2 !!!')  # !
        if self.alarm_arc_idx_s > 0:
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_s - self.af_win_size:self.alarm_arc_idx_s] = \
                [self.indicator_max_val / 4 * 1] * self.af_win_size
        self.last_peak_idx = peak_idx
        # self.last_peak_val = _peak_val
        # self.seq_power_proc_len = -self.af_win_size * 1.0  # !

    def infer_v5(self, feat_sample=False):
        self.seq_power_proc_len += 1
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        _alarm_arc_cnt_th = 1.5
        _min_val_th = self.indicator_max_val / 2 * 0.05  # MIN_VAL_TH * 2
        _th_raw = 2048 * 1.0
        _ini_peak_cnt_th = 64
        if self.alarm_idle_cnt > 0:
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val
        if self.ini_peak_cnt < _ini_peak_cnt_th:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 4
        if self.peak_miss_cnt > self.af_win_size * 8:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val / 8
            self.ini_peak_cnt = 0
        if self.seq_power_proc_len < self.af_win_size:  # !
            return
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        if self.peak_miss_cnt > self.af_win_size * 4 and self.alarm_arc_idx_e < 0 < self.alarm_arc_idx_s:
            self.alarm_arc_idx_e = self.db.db['rt'].seq_len
            logging.info(f'end v1 -- > {self.alarm_arc_cnt}')
        _alarm_indicate_scale = ALARM_INDICATE_SCALE
        if self.alarm_idle_cnt > 0 and self.ini_peak_cnt == 0:  # idle alarm confirm
            print('arc fault alarm v0 !!!')
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val * _alarm_indicate_scale
        if self.alarm_arc_idx_e > 0 and self.alarm_arc_idx_s > 0:
            # self.db.db['rt'].seq_power[self.alarm_arc_idx_e - 1] = self.indicator_max_val * 1.5  # affect demo_pico !
            # self.db.db['rt'].info_pred_peaks.append(self.alarm_arc_idx_e - 1)
            # self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                [self.indicator_max_val / 8 * 1] * self.af_win_size
            if self.alarm_arc_cnt > _alarm_arc_cnt_th:
                self.alarm_arc_state = 0
                if self.ini_peak_cnt < _ini_peak_cnt_th:  # idle state alarm
                    self.alarm_idle_cnt = 8
                else:
                    print('arc fault alarm v1 !!!')
                    self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_e - self.af_win_size:self.alarm_arc_idx_e] = \
                        [self.indicator_max_val * _alarm_indicate_scale] * self.af_win_size
            self.alarm_arc_idx_s, self.alarm_arc_idx_e = -1, -1

        # power_pick = self.db.db['rt'].seq_power[-1]
        # if self.db.db['rt'].seq_len == 458:
        #     print("manu")
        # self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
        #     else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        # self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + _min_val_th)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        # if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 1.5:  # seq filter
        if peak_idx < 0:  # seq filter
            # self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            # self.alarm_lower_cnt = self.alarm_lower_cnt - 0.0001 if self.alarm_lower_cnt > 0 else self.alarm_lower_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.alarm_idle_cnt = self.alarm_idle_cnt - 0.0005 if self.alarm_idle_cnt > 0 else self.alarm_idle_cnt
            self.peak_miss_cnt += 1
            return
        self.ini_peak_cnt += 1
        self.peak_anchor_idx = peak_idx
        self.peak_miss_cnt = 0
        if self.ini_peak_cnt < 2:  # !
            return
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _seq_pick = _seq_pick_power
        self.db.db['rt'].info_pred_peaks.append(peak_idx)
        _data = _seq_pick[np.newaxis, :]
        _score, _feat = self.classifier.infer(_data, batch_size=1)
        _score = _score[0]
        # if _score * self.indicator_max_val > 30:
        #     print("manu")
        self.db.db['rt'].seq_state_pred_classifier[peak_idx - self.af_win_size:peak_idx] = \
            [_score * self.indicator_max_val] * self.af_win_size
        _alarm_arc_th = _th_raw / self.indicator_max_val
        _is_arc = _score * self.indicator_max_val > _th_raw
        # self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt - 0.5 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt  # !
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc and _peak_val > self.indicator_max_val * 0.9 else self.alarm_arc_cnt  # !
        # if _is_arc and _peak_val > self.indicator_max_val * 0.9:
        #     print("2")
        # elif _is_arc:
        #     print("1")
        if self.alarm_arc_cnt > 0:
            logging.info(f'self.alarm_arc_cnt -- > {self.alarm_arc_cnt}')
        self.db.db['rt'].info_af_scores.append(self.alarm_arc_cnt)
        self.alarm_arc_idx_s = peak_idx if self.alarm_arc_idx_s < 0 and _is_arc and self.alarm_idle_cnt <= 0 \
            else self.alarm_arc_idx_s
        if self.alarm_arc_idx_s > 0 and not _is_arc and self.alarm_idle_cnt <= 0:
            self.alarm_arc_idx_e = peak_idx
            logging.info(f'end v0 -- > {self.alarm_arc_cnt}')
        if (self.alarm_arc_cnt > _alarm_arc_cnt_th
                and self.db.db['rt'].seq_state_pred_arc[peak_idx - 1] < self.indicator_max_val):  # pre-alarm
            self.db.db['rt'].seq_state_pred_arc[peak_idx - self.af_win_size:peak_idx] = \
                [self.indicator_max_val / 4 * 3] * self.af_win_size
            # print('arc fault alarm v2 !!!')  # !
        if self.alarm_arc_idx_s > 0:
            self.db.db['rt'].seq_state_pred_arc[self.alarm_arc_idx_s - self.af_win_size:self.alarm_arc_idx_s] = \
                [self.indicator_max_val / 4 * 1] * self.af_win_size
        self.last_peak_idx = peak_idx
        # self.last_peak_val = _peak_val
        self.seq_power_proc_len = -self.af_win_size * 1.0  # !

    def infer_fpga(self, feat_sample=False):
        """
        self.indicator_max_val --> 4096
        """
        _alarm_arc_cnt_th = 4
        _min_val_th = self.indicator_max_val / 2 * 0.05  # MIN_VAL_TH * 2
        _th_raw = 2048 * 1.0
        _ini_peak_cnt_th = 64
        _is_alarm_overload = self.alarm_overload_cnt > _alarm_arc_cnt_th

        # alarm evaluation
        if _is_alarm_overload:
            self.db.db['rt'].seq_state_pred_idle[-1] = self.indicator_max_val
        _alarm_indicate_scale = 1.5
        if self.alarm_arc_cnt > _alarm_arc_cnt_th:
            print('arc fault alarm !!!')
            self.db.db['rt'].seq_state_pred_arc[-1] = self.indicator_max_val * _alarm_indicate_scale

        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + _min_val_th)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0 or peak_idx - self.last_peak_idx < self.af_win_size // 1.5:  # seq filter
            self.alarm_overload_cnt = self.alarm_overload_cnt - 0.0001 if self.alarm_overload_cnt > 0 else self.alarm_overload_cnt
            self.alarm_lower_cnt = self.alarm_lower_cnt - 0.0001 if self.alarm_lower_cnt > 0 else self.alarm_lower_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt - 0.0001 if self.alarm_arc_cnt > 0 else self.alarm_arc_cnt
            self.alarm_arc_cnt = self.alarm_arc_cnt + 0.0001 if self.alarm_arc_cnt < 0 else self.alarm_arc_cnt
            self.alarm_idle_cnt = self.alarm_idle_cnt - 0.0001 if self.alarm_idle_cnt > 0 else self.alarm_idle_cnt
            return
        self.peak_anchor_idx = peak_idx
        _peak_val = self.db.db['rt'].seq_power[peak_idx]
        self.alarm_overload_cnt = self.alarm_overload_cnt + 1 if _peak_val > self.indicator_max_val else self.alarm_overload_cnt
        _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
        _th_arc = (_peak_val - self.power_mean) * 0.03
        _cnt_arc = np.sum(np.abs(_seq_pick_power - self.power_mean) < _th_arc)
        _scale_arc = 128
        _cnt_arc = _cnt_arc if _cnt_arc * _scale_arc < 4096 else 4096 / _scale_arc
        self.db.db['rt'].seq_state_pred_balcony[peak_idx - self.af_win_size:peak_idx] = \
            [_cnt_arc * _scale_arc] * self.af_win_size
        _is_arc = _cnt_arc * _scale_arc > _th_raw
        self.alarm_arc_cnt = self.alarm_arc_cnt + 1 if _is_arc else self.alarm_arc_cnt
        self.last_peak_idx = peak_idx

    def _preprocess(self):
        # filtered_sample, self.filter_zi = self._realtime_highpass_filter(self.db.db['rt'].seq_power[-1])
        # self.db.db['rt'].seq_filtered[-1] = filtered_sample
        # if self.sub_sample_cnt == self.sub_sample_rate:
        #     self.db.db['rt'].seq_power_ss.append(filtered_sample)
        if self.sub_sample_cnt == self.sub_sample_rate:
            self.db.db['rt'].seq_power_ss.append(self.db.db['rt'].seq_power[-1])
        self.sub_sample_cnt = self.sub_sample_cnt + 1 if self.sub_sample_cnt < self.sub_sample_rate else 1

    def _peaks_sample(self, peak_idx, adjustment_range=64, n_aug=32):
        for i in range(n_aug):  # Generate 16 positive samples
            # Randomly adjust peak_idx within the range of ±64
            random_adjustment = random.randint(-adjustment_range, adjustment_range)
            adjusted_peak_idx = \
                max(self.af_win_size, min(peak_idx + random_adjustment, self.db.db['rt'].seq_len - 1))
            adjusted_peak_idx = peak_idx if i == 0 else adjusted_peak_idx
            _seq_pick_power = np.array(
                self.db.db['rt'].seq_power[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(float)
            _seq_pick_hf = np.array(
                self.db.db['rt'].seq_hf[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(float)
            _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
            self.db.db['rt'].seq_state_pred_arc[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx] = \
                [self.indicator_max_val / 2] * self.af_win_size
            self.samples_pos.append(_seq_pick)
            self.db.db['rt'].info_pred_peaks.append(adjusted_peak_idx)
            self.db.db['rt'].info_af_scores.append(0.)

    def sample_ae(self):
        """
        Sample without peak detection based on label at the end of the window.
        Saves a sequence as a positive sample if its end index is in a positive state,
        and as a negative sample otherwise.
        """
        # Wait until we have enough data for a full window
        if self.db.db['rt'].seq_len < self.af_win_size:
            return

        # Define indices
        end_idx_exclusive = self.db.db['rt'].seq_len
        start_idx = end_idx_exclusive - self.af_win_size
        label_idx = end_idx_exclusive - 1

        # Get label from the end of the window
        is_positive = self.db.db['rt'].seq_state_gt_arc[label_idx] > 0

        # Extract the sequence from seq_power
        _seq_pick = np.array(self.db.db['rt'].seq_power[start_idx:end_idx_exclusive]).astype(float)

        # Do not save if peak-to-peak amplitude is too small
        if np.max(_seq_pick) - np.min(_seq_pick) <= MIN_VAL_TH * 2:
            return

        # Save the sample
        (self.samples_pos if is_positive else self.samples_neg).append(_seq_pick)
        if is_positive:
            self.db.db['rt'].seq_state_pred_arc[label_idx] = self.indicator_max_val / 2

    def sample(self, pos_only=False):
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + MIN_VAL_TH)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:  # seq filter
            return
        if self.db.db['rt'].seq_state_gt_arc[peak_idx] > 0:  # peak in the range
            self._peaks_sample(peak_idx)
        elif not pos_only:
            _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
            # _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
            # _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
            _seq_pick = _seq_pick_power
            _seq_pick_ex = _seq_pick[np.newaxis, :]
            # _score = self.classifier.infer(_seq_pick_ex, batch_size=1)[0]
            _score = 0.0
            if self.db.db['rt'].seq_power[
                peak_idx] < self.indicator_max_val * ALARM_INDICATE_SCALE:  # ignore artificial noise
                self.samples_neg.append(_seq_pick)  # sample all
                self.db.db['rt'].info_pred_peaks.append(peak_idx)
                self.db.db['rt'].info_af_scores.append(0.)
            if _score > 1.0:  # set 1.0 to disable
                self.samples_neg.append(_seq_pick)
                for _ in range(32):
                    # Randomly adjust peak_idx within the range of ±64
                    adjustment_range = 64
                    random_adjustment = random.randint(-adjustment_range, adjustment_range)
                    adjusted_peak_idx = \
                        max(self.af_win_size, min(peak_idx + random_adjustment, self.db.db['rt'].seq_len - 1))
                    # adjusted_peak_idx = peak_idx
                    _seq_pick_power = np.array(
                        self.db.db['rt'].seq_power[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(
                        float)
                    _seq_pick_hf = np.array(
                        self.db.db['rt'].seq_hf[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(float)
                    _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
                    self.db.db['rt'].seq_state_pred_arc[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx] = \
                        [self.indicator_max_val / 2] * self.af_win_size
                    self.samples_neg.append(_seq_pick)
                    self.db.db['rt'].info_pred_peaks.append(adjusted_peak_idx)
                    self.db.db['rt'].info_af_scores.append(0.)

    def sample_pos_v0(self):
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + MIN_VAL_TH)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:  # seq filter
            return
        if self.db.db['rt'].seq_power[peak_idx] > self.indicator_max_val * 0.9:
            self._peaks_sample(peak_idx, n_aug=1)  # no augmentation
        else:  # add negs
            _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
            _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
            _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
            self.samples_neg.append(_seq_pick)
            self.db.db['rt'].info_pred_peaks.append(peak_idx)
            self.db.db['rt'].info_af_scores.append(0.)

    def sample_neg_v0(self):
        if self.db.db['rt'].seq_len < self.af_win_size:  # waiting for enough data
            return
        power_pick = self.db.db['rt'].seq_power[-1]
        self.power_mean = self.power_mean * (1 - self.pm_lr) + power_pick * self.pm_lr if self.power_mean > 0 \
            else (np.max(self.db.db['rt'].seq_power) + np.min(self.db.db['rt'].seq_power)) / 2.
        self.db.db['rt'].seq_power_mean[-1] = self.power_mean
        peak_idx_norm = self._detect_peak(power_pick, win_size=self.peak_eval_win_size,
                                          peak_th=self.power_mean + MIN_VAL_TH)
        peak_idx = self.db.db['rt'].seq_len - self.peak_eval_win_size // 2 if peak_idx_norm > 0 else -1
        if peak_idx < 0:  # seq filter
            return
        adjustment_range = 64
        if not self.db.db['rt'].seq_state_gt_arc[peak_idx] > 0:  # peak in the range
            _seq_pick_power = np.array(self.db.db['rt'].seq_power[peak_idx - self.af_win_size:peak_idx]).astype(float)
            _seq_pick_hf = np.array(self.db.db['rt'].seq_hf[peak_idx - self.af_win_size:peak_idx]).astype(float)
            _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
            _seq_pick_ex = _seq_pick[np.newaxis, :]
            _score = self.classifier.infer(_seq_pick_ex, batch_size=1)[0]
            if _score > 0.3:
                self.samples_neg.append(_seq_pick)
                for _ in range(4):
                    # Randomly adjust peak_idx within the range of ±64
                    random_adjustment = random.randint(-adjustment_range, adjustment_range)
                    adjusted_peak_idx = \
                        max(self.af_win_size, min(peak_idx + random_adjustment, self.db.db['rt'].seq_len - 1))
                    # adjusted_peak_idx = peak_idx
                    _seq_pick_power = np.array(
                        self.db.db['rt'].seq_power[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(
                        float)
                    _seq_pick_hf = np.array(
                        self.db.db['rt'].seq_hf[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx]).astype(float)
                    _seq_pick = np.concatenate((_seq_pick_power, _seq_pick_hf), axis=0)
                    self.db.db['rt'].seq_state_pred_arc[adjusted_peak_idx - self.af_win_size:adjusted_peak_idx] = \
                        [self.indicator_max_val / 2] * self.af_win_size
                    self.samples_neg.append(_seq_pick)
                    self.db.db['rt'].info_pred_peaks.append(adjusted_peak_idx)
                    self.db.db['rt'].info_af_scores.append(0.)
