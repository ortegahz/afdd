import argparse
import logging

from cores.arc_detector import ArcDetector
from data.data import DataV0
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addr', default='/media/manu/data/afdd/故障电弧实验/并联碳化3KW_0.7')
    return parser.parse_args()


def run(args):
    logging.info(args)
    db_offline = DataV0(args.addr)
    db_offline.load()
    arc_detector = ArcDetector()
    for key in db_offline.db.keys():
        db_offline_single = db_offline.db[key]
        for idx in range(db_offline_single.len):
            cur_power = db_offline_single.seq_power[idx]
            cur_state_gt_arc = db_offline_single.seq_state_arc[idx]
            cur_state_gt_normal = db_offline_single.seq_state_normal[idx]
            arc_detector.db.update(cur_power=cur_power,
                                   cur_state_gt_arc=cur_state_gt_arc,
                                   cur_state_gt_normal=cur_state_gt_normal)
        arc_detector.db.plot()
        arc_detector.db.reset()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
