# FILE: detector_wrapper.py

from cores.arc_detector import ArcDetector
from data.data import *
from utils.utils import make_dirs


class DetectorWrapperBase:
    def __init__(self, addr, dir_save=None):
        self.addr = addr
        self.dir_save = dir_save
        self.pause_time_s = 4096
        self.plot_show = True
        make_dirs(dir_save, reset=True)

    def run(self):
        raise NotImplementedError


class DetectorWrapperV0(DetectorWrapperBase):
    """
    format: case/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save=None, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save)
        # self.pause_time_s = 0.01
        # self.plot_show = False
        self.key_pick = key_pick
        self.arc_detector = ArcDetector()
        self.dbo_type = dbo_type

        # --- Configurable saving options ---
        self.svm_label_file = '/media/manu/ST8000DM004-2U91/tmp/afd'
        self.save_as_svm, self.save_as_h5 = False, False
        self.h5_path = '/media/manu/ST8000DM004-2U91/tmp/afd.h5'

        if self.save_as_svm and os.path.exists(self.svm_label_file):
            os.remove(self.svm_label_file)

        if self.save_as_h5 and self.h5_path and os.path.exists(self.h5_path):
            os.remove(self.h5_path)

    def _process_single(self, key, case_name, feat_sample=False, blacklist_sample=False, is_infer=True):
        db_offline_single = self.db_offline.db[key]
        # for idx in range(0, db_offline_single.len, self.arc_detector.sub_sample_rate):
        for idx in range(0, db_offline_single.len):
            # if idx < 0.36 * 1e6 or idx >= 0.38 * 1e6:  # for mcu alg test
            #     continue
            # if idx < 0.07 * 1e6 or idx >= 0.09 * 1e6:  # for mcu alg test
            #     continue
            cur_power = db_offline_single.seq_power[idx]
            cur_hf = db_offline_single.seq_hf[idx]
            cur_state_gt_arc = db_offline_single.seq_state_arc[idx]
            cur_state_gt_normal = db_offline_single.seq_state_normal[idx]
            cur_power_voltage = db_offline_single.seq_power_voltage[idx]
            self.arc_detector.db.update(cur_power=cur_power,
                                        cur_hf=cur_hf,
                                        cur_state_gt_arc=cur_state_gt_arc,
                                        cur_state_gt_normal=cur_state_gt_normal,
                                        cur_power_voltage=cur_power_voltage)
            if is_infer:
                # self.arc_detector.infer_v6(feat_sample=feat_sample, blacklist_sample=blacklist_sample)
                self.arc_detector.infer_v5(feat_sample=feat_sample)  # TAG: for mcu
                # self.arc_detector.infer_v3(feat_sample=feat_sample)
            if self.save_as_svm:
                self.arc_detector.sample()
                # self.arc_detector.sample_ae()
                # self.arc_detector.sample(pos_only=True)
                # self.arc_detector.sample_pos_v0()
        if is_infer:
            self.arc_detector.db.plot(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
                                      save_name=f'{case_name}.png', show=self.plot_show)
            # self.arc_detector.db.plot_arc(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
            #                               save_name=f'{case_name}.png', show=self.plot_show)
            # self.arc_detector.db.plot_arc_neg(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
            #                                   save_name=f'{case_name}.png', show=self.plot_show)
            # self.arc_detector.db.plot_cwt(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
            #                           save_name=f'{case_name}_{key}.png', show=self.plot_show)
            # self.arc_detector.db.plot_emd(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
            #                               save_name=f'{case_name}_{key}.png', show=self.plot_show)
        if self.save_as_svm:
            self.arc_detector.save_samples(path_save=self.svm_label_file)

        if self.save_as_h5:
            self.arc_detector.save_to_hdf5(path_save=self.h5_path)

        # self.arc_detector.save_seq()
        # self.arc_detector.db.save()
        if self.arc_detector.is_alarm:
            self.arc_detector.alarm_seq_lst.append(case_name)
        self.arc_detector.reset()

    def run(self):
        self.db_offline = eval(self.dbo_type)(self.addr)
        # self.db_offline = DataV0(addr)
        self.db_offline.load()
        case_name = os.path.basename(self.addr)
        for key in self.db_offline.db.keys():
            if self.key_pick is not None and key != self.key_pick:
                continue
            self._process_single(key, case_name)


class DetectorWrapperV1(DetectorWrapperV0):
    """
    format: dir/<cases>/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.pause_time_s = 0.01
        self.plot_show = False
        # self.arc_detector = ArcDetector()

    def run(self):
        _cnt = 0
        cases_dir = glob.glob(os.path.join(self.addr, '*'))
        for i, case_dir in enumerate(cases_dir):
            logging.info(f'[{len(cases_dir)}] {i}th case_dir: {case_dir}')
            case_name = os.path.basename(case_dir)
            logging.info(f'case_name: {case_name}')
            self.db_offline = DataV0(case_dir)
            self.db_offline.load()
            for key in self.db_offline.db.keys():
                self._process_single(key, f'{_cnt}_' + case_name)
                _cnt += 1


class DetectorWrapperV1NPY(DetectorWrapperV1):
    """
    format: dir/<npys> + <txts>
    """

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.arc_detector = ArcDetector()

    def run(self):
        _cnt = 0
        cases_path = glob.glob(os.path.join(self.addr, '*.npy'))
        for i, case_path in enumerate(cases_path):
            logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
            # case_name, _ = os.path.basename(case_path).split('.')
            case_name, _ = os.path.splitext(os.path.basename(case_path))
            self.db_offline = DataV4(case_path)
            self.db_offline.load()
            for key in self.db_offline.db.keys():
                self._process_single(key, f'{_cnt}_' + case_name)
                _cnt += 1


class DetectorWrapperV2(DetectorWrapperV1):
    """
    format: <cases_type>/<cases>/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.pause_time_s = 0.01
        self.plot_show = False
        # self.arc_detector = ArcDetector()

    def run(self):
        _cnt = 0
        cases_types_dir = glob.glob(os.path.join(self.addr, '*'))
        for i, cases_type_dir in enumerate(cases_types_dir):
            logging.info(f'[{len(cases_types_dir)}] {i}th cases_type_dir: {cases_type_dir}')
            cases_dir = glob.glob(os.path.join(cases_type_dir, '*'))
            for j, case_dir in enumerate(cases_dir):
                logging.info(f'[{len(cases_dir)}] {j}th case_dir: {case_dir}')
                case_name = os.path.basename(case_dir)
                # logging.info(f'case_name: {case_name}')
                self.db_offline = DataV0(case_dir)
                self.db_offline.load()
                for key in self.db_offline.db.keys():
                    self._process_single(key, f'{_cnt}_' + case_name)
                    _cnt += 1


class DetectorWrapperV2NPY(DetectorWrapperV2):
    """
    format: dir/<cases_dir>/<.npys> + <.txts>
    """

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.arc_detector = ArcDetector()

    def run(self):
        _cnt = 0
        cases_types_dir = glob.glob(os.path.join(self.addr, '*'))
        for i, cases_type_dir in enumerate(cases_types_dir):
            logging.info(f'[{len(cases_types_dir)}] {i}th cases_type_dir: {cases_type_dir}')
            cases_path = glob.glob(os.path.join(cases_type_dir, '*.npy'))
            for i, case_path in enumerate(cases_path):
                logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
                # case_name, _ = os.path.basename(case_path).split('.')
                case_name, _ = os.path.splitext(os.path.basename(case_path))
                logging.info(f'case_name: {case_name}')
                self.db_offline = DataV4(case_path)
                self.db_offline.load()
                for key in self.db_offline.db.keys():
                    self._process_single(key, f'{_cnt}_' + case_name)
                    _cnt += 1


class DetectorWrapperV3NPY(DetectorWrapperV2):
    """
    format: dir/<any> + <.npys> + <.txts>/...
    """

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.pause_time_s = 0.01
        self.plot_show = False
        # self.arc_detector = ArcDetector()

    def run(self, _feat_sample=False, _blacklist_sample=False):
        _cnt = 0
        cases_path = glob.glob(os.path.join(self.addr, '**', '*.npy'), recursive=True)
        for i, case_path in enumerate(cases_path):
            # if _cnt <= 10:
            #     _cnt += 1
            #     continue
            # _feat_sample = True if _cnt == 0 else False
            logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
            case_name, _ = os.path.splitext(os.path.basename(case_path))
            logging.info(f'case_name: {case_name}')
            self.db_offline = DataV4(case_path)
            self.db_offline.load()
            for key in self.db_offline.db.keys():
                self._process_single(key, f'{_cnt}_' + case_name, feat_sample=_feat_sample,
                                     blacklist_sample=_blacklist_sample)
                _cnt += 1


class DetectorWrapperV3BIN(DetectorWrapperV3NPY):

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV0'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.save_as_svm, self.save_as_h5 = False, False
        self.is_infer = not self.save_as_svm  # must be

        if self.save_as_svm and os.path.exists(self.svm_label_file):
            os.remove(self.svm_label_file)

        if self.save_as_h5 and self.h5_path and os.path.exists(self.h5_path):
            os.remove(self.h5_path)

    def run(self, _feat_sample=True, _blacklist_sample=False):
        _cnt = 0
        cases_path = glob.glob(os.path.join(self.addr, '**', '*.bin'), recursive=True)
        for i, case_path in enumerate(cases_path):
            # if _cnt <= 10:
            #     _cnt += 1
            #     continue
            # _feat_sample = True if _cnt == 0 else False
            logging.info(f'[{len(cases_path)}] {i}th case_path: {case_path}')
            case_name, _ = os.path.splitext(os.path.basename(case_path))
            logging.info(f'case_name: {case_name}')
            # self.db_offline = DataV5(case_path)
            self.db_offline = eval(self.dbo_type)(case_path)
            self.db_offline.load()
            for key in self.db_offline.db.keys():
                self._process_single(key, f'{_cnt}_' + case_name, feat_sample=_feat_sample,
                                     blacklist_sample=_blacklist_sample, is_infer=self.is_infer)
                _cnt += 1

        print(f"alarm seq cnt --> {len(self.arc_detector.alarm_seq_lst)} | {_cnt} {self.arc_detector.alarm_seq_lst}")

        if _feat_sample:
            self.arc_detector.save_feats_ref()
        if _blacklist_sample:
            self.arc_detector.save_feats_ref_blacklist()


class DetectorWrapperV4H5(DetectorWrapperV3BIN):

    def __init__(self, addr, dir_save, key_pick=None, dbo_type='DataV6'):
        super().__init__(addr, dir_save, key_pick=key_pick, dbo_type=dbo_type)
        self.save_as_h5 = True
        self.save_as_svm = False

    def run(self):
        if not os.path.exists(self.addr):
            logging.error(f"List file not found: {self.addr}")
            return

        with open(self.addr, 'r') as f:
            dir_list = [line.strip() for line in f.readlines() if line.strip()]

        _cnt = 0
        for dir_path in dir_list:
            logging.info(f"Scanning directory: {dir_path}")

            folder_name = os.path.basename(dir_path.rstrip(os.sep))
            self.h5_path = os.path.join(dir_path, f"{folder_name}.h5")

            if os.path.exists(self.h5_path):
                os.remove(self.h5_path)

            cases_path = glob.glob(os.path.join(dir_path, '**', '*.bin'), recursive=True)

            for idx, case_path in enumerate(cases_path):
                logging.info(f'[{idx} / {len(cases_path)}] Processing case: {case_path}')
                try:
                    self.db_offline = eval(self.dbo_type)(case_path)
                    self.db_offline.load()

                    for key in self.db_offline.db.keys():
                        self._process_single(key, f'{_cnt}_' + folder_name, is_infer=False)
                        _cnt += 1
                except Exception as e:
                    logging.error(f"Failed to process {case_path}: {e}")
