import glob
import logging
import os

from cores.arc_detector import ArcDetector
from data.data import DataV0
from utils.utils import make_dirs


class DetectorWrapperBase:
    def __init__(self, addr, dir_save=None):
        self.addr = addr
        self.dir_save = dir_save
        self.pause_time_s = 1024
        self.plot_show = True
        make_dirs(dir_save, reset=True)

    def run(self):
        raise NotImplementedError


class DetectorWrapperV0(DetectorWrapperBase):
    """
    format: case/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save=None, key_pick=None, svm_label_file='/home/manu/tmp/smartsd'):
        super().__init__(addr, dir_save)
        # self.pause_time_s = 0.01
        # self.plot_show = False
        self.key_pick = key_pick
        self.arc_detector = ArcDetector()
        self.db_offline = DataV0(addr)
        self.db_offline.load()
        self.svm_label_file = svm_label_file
        if os.path.exists(self.svm_label_file):
            os.remove(self.svm_label_file)

    def _process_single(self, key, case_name):
        db_offline_single = self.db_offline.db[key]
        for idx in range(db_offline_single.len):
            cur_power = db_offline_single.seq_power[idx]
            cur_hf = db_offline_single.seq_hf[idx]
            cur_state_gt_arc = db_offline_single.seq_state_arc[idx]
            cur_state_gt_normal = db_offline_single.seq_state_normal[idx]
            self.arc_detector.db.update(cur_power=cur_power,
                                        cur_hf=cur_hf,
                                        cur_state_gt_arc=cur_state_gt_arc,
                                        cur_state_gt_normal=cur_state_gt_normal)
            self.arc_detector.infer_v2()
            # self.arc_detector.sample()
        self.arc_detector.db.plot(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
                                  save_name=f'{case_name}_{key}.png', show=self.plot_show)
        # self.arc_detector.db.plot_cwt(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
        #                           save_name=f'{case_name}_{key}.png', show=self.plot_show)
        # self.arc_detector.db.plot_emd(pause_time_s=self.pause_time_s, dir_save=self.dir_save,
        #                               save_name=f'{case_name}_{key}.png', show=self.plot_show)
        self.arc_detector.save_samples(path_save=self.svm_label_file)
        self.arc_detector.reset()

    def run(self):
        case_name = os.path.basename(self.addr)
        for key in self.db_offline.db.keys():
            if self.key_pick is not None and key != self.key_pick:
                continue
            self._process_single(key, case_name)


class DetectorWrapperV1(DetectorWrapperV0):
    """
    format: dir/<cases>/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save, key_pick=None):
        super().__init__(addr, dir_save, key_pick=key_pick)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.arc_detector = ArcDetector()

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


class DetectorWrapperV2(DetectorWrapperV1):
    """
    format: <cases_type>/<cases>/<key_*>.BIN + ... + *.xlsx
    """

    def __init__(self, addr, dir_save, key_pick=None):
        super().__init__(addr, dir_save, key_pick=key_pick)
        self.pause_time_s = 0.01
        self.plot_show = False
        self.arc_detector = ArcDetector()

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
