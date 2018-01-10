#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import itertools
import os
import pickle
import numpy as np

from antlia.record import load_file
from antlia import plot_braking as braking


def __get_record_files(extension):
    record_path = os.path.join(os.path.dirname(__file__),
                               r'../../data/etrike/experiment/')
    rider_path = sorted(glob.glob(os.path.join(record_path, r'rider*/')))
    files = []
    for rp in rider_path:
        trial_path = sorted(glob.glob(
            os.path.join(rp,
                         r'convbike/*{}'.format(extension))))
        files.append(trial_path)
    return files

BICYCLE_RECORD_FILES = __get_record_files('.csv')
assert BICYCLE_RECORD_FILES, "No bicycle records found!"

def _get_bicycle_records():
    # load calibration data
    calibration_path = os.path.join(os.path.dirname(__file__),
                                    '..', r'config.p')
    with open(calibration_path, 'rb') as f:
        calibration = pickle.load(f)

    trials = []
    for rid, rider_path in enumerate(BICYCLE_RECORD_FILES):
        for tid, trial_path in enumerate(rider_path):
            try:
                trial = load_file(trial_path, calibration['convbike'])
            except IndexError:
                continue
            trials.append((rid, tid, trial))
    return trials

def load_records():
    return _get_bicycle_records()

def get_metrics(records):
    metrics = []
    for rid, tid, trial in records:
        try:
            m, _, _, _ = braking.get_metrics(trial)
            # rider id and trial id aren't available within the record
            # datatype so we need to add them here
            m['rider id'] = rid
            m['trial id'] = tid
            metrics.append(m)
        except TypeError:
            continue
    return np.concatenate(metrics)
