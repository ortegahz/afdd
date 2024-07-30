import argparse
import logging
from multiprocessing import Process, Queue, Event

from cores.pico import data_acquisition_process, data_plotting_process, afdd_process
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def run(args):
    logging.info(args)
    data_queue = Queue()
    overflow_queue = Queue()
    stop_event = Event()
    acquisition_process = Process(target=data_acquisition_process, args=(data_queue, overflow_queue, stop_event))
    # _process = Process(target=data_plotting_process, args=(data_queue, overflow_queue, stop_event))
    _process = Process(target=afdd_process, args=(data_queue, overflow_queue, stop_event))
    acquisition_process.start()
    _process.start()
    acquisition_process.join()
    _process.join()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
