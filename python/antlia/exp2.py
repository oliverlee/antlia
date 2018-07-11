#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

from antlia import dtype
from antlia import record
from antlia import trial2

BICYCLE_LOG_FILES = [
    '2018-04-23_12-30-38.csv',
    '2018-04-23_13-13-36.csv',
    '2018-04-23_14-22-58.csv',
    '2018-04-23_15-27-48.csv',
    '2018-04-23_16-32-27.csv',
    '2018-04-23_17-14-00.csv',
    '2018-04-25_09-27-24.csv',
    '2018-04-25_10-20-28.csv',
    '2018-04-25_11-34-04.csv',
    '2018-04-25_12-41-48.csv',
    '2018-04-25_14-14-57.csv',
    '2018-04-25_14-49-39.csv',
    '2018-04-25_16-15-57.csv',
    '2018-04-25_17-23-04.csv',
    '2018-04-26_11-19-31.csv',
    '2018-04-26_14-50-53.csv',
    '2018-04-27_14-59-52.csv'
]

LIDAR_LOG_FILES = [
    '2018-04-23-12-17-37_0.pkl.gz',
    '2018-04-23-13-01-00_0.pkl.gz',
    '2018-04-23-14-10-33_0.pkl.gz',
    '2018-04-23-15-15-14_0.pkl.gz',
    '2018-04-23-16-19-35_0.pkl.gz',
    '2018-04-23-17-01-24_0.pkl.gz',
    '2018-04-25-09-15-00_0.pkl.gz',
    '2018-04-25-10-07-31_0.pkl.gz',
    '2018-04-25-11-21-29_0.pkl.gz',
    '2018-04-25-12-29-06_0.pkl.gz',
    '2018-04-25-14-02-15_0.pkl.gz',
    '2018-04-25-14-36-55_0.pkl.gz',
    '2018-04-25-16-03-24_0.pkl.gz',
    '2018-04-25-17-10-07_0.pkl.gz',
    '2018-04-26-11-07-38_0.pkl.gz',
    '2018-04-26-14-38-03_0.pkl.gz',
    '2018-04-27-14-47-07_0.pkl.gz',
    '2018-04-27-15-39-56_0.pkl.gz'
]

MISSING_SYNC = [
   [680],
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None,
   None
]

TRIAL_MASK = [
   None,
   None,
   0,
   None,
   None,
   0,
   None,
   None,
   9,
   None,
   None,
   11,
   8,
   9,
   None,
   None,
   None
]

# one less bicycle log recorded due to logger crash
assert len(BICYCLE_LOG_FILES) == len(LIDAR_LOG_FILES) - 1
assert len(MISSING_SYNC) == len(TRIAL_MASK)
assert len(BICYCLE_LOG_FILES) == len(MISSING_SYNC)


def load_records(index=None, data_dir=None):
    """Load the experiment records from Gothenburg April 2018. Notes on missing
    synchronizations and repeated trials are applied.

    Parameters:
    index: int or slice or index array or None, valid numpy array index to
           specify which records to load. A value of None will load all records.
           Defaults to None.
    data_dir: Specifies path to data directory containing bicycle and lidar log
              files. The lidar log files must be pre-processed as this function
              does not handle bag files.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__),
                                r'../../data/comfort/')

    # use hardcoded config file
    with open('../config.p', 'rb') as f:
        bicycle_calibration = pickle.load(f)

    exp_records = []
    exp_index = np.arange(len(BICYCLE_LOG_FILES))

    if index is not None:
        exp_index = exp_index[index]

    for i in exp_index:
        # create record from lidar and bicycle logs
        r = record.Record(
            dtype.load_converted_record(
                os.path.join(data_dir, LIDAR_LOG_FILES[i])),
            record.load_file(
                os.path.join(data_dir, BICYCLE_LOG_FILES[i]),
                              bicycle_calibration['convbike']))

        # synchronize records and detect trials
        r.sync()
        r._calculate_trials2(missing_sync=MISSING_SYNC[i],
                             trial_mask=TRIAL_MASK[i])

        # append processed record
        exp_records.append(r)

    return exp_records


def instructed_speed(record_id, trial_id):
    """Return instructed speed for experiment trial.
    """
    speed_order = np.r_[12, 17, 22,
                        12, 22, 17,
                        17, 12, 22,
                        17, 22, 12,
                        22, 12, 17,
                        22, 17, 12].astype(np.float) / 3.6
    return np.roll(speed_order, -3*record_id)[trial_id]
