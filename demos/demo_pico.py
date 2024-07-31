import argparse
import logging
from multiprocessing import Process, Queue, Event

import keyboard

from cores.pico import data_acquisition_process, afdd_process
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
    acquisition_process.start()

    processing_process = Process(target=afdd_process, args=(data_queue, overflow_queue, stop_event))
    processing_process.start()

    def save_data_and_exit():
        logging.info("Key pressed. Setting stop event...")
        stop_event.set()

    keyboard.add_hotkey('q', save_data_and_exit)

    acquisition_process.join()
    processing_process.join()


def main():
    set_logging()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
