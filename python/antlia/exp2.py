#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
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
   [9, 10],
   None,
   None,
   11,
   8,
   9,
   None,
   None,
   None
]

## one less bicycle log recorded due to logger crash
assert len(BICYCLE_LOG_FILES) == len(LIDAR_LOG_FILES) - 1
assert len(MISSING_SYNC) == len(TRIAL_MASK)
assert len(BICYCLE_LOG_FILES) == len(MISSING_SYNC)

## trial specific bounding box masks for the following trials:
## 1, 2
## 2, 4
## 3, 12
## 3, 14
##
## time for trial (1, 2) determined from:
## >> x, y, t = records[1].trials[2].lidar.cartesianz(**trial2.VALID_BB)
## >> FALL_BBOX = {
## >>     'xlim': (-5, -2),
## >>     'ylim': (2.7, 2.8),
## >> }
## >> mask = trial2._apply_bbmask(FALL_BBOX, x, y, apply_mask=False)
## >> t0 = t[mask][0]
## >> print (t0, t.max() + 1)
## 291.103905916 325.454327583

TRIAL_BBMASK = {
    #disable 1-2 trial bbmask
    #(1, 2): {
    #    'xlim': (-5, -2.5),
    #    'ylim': (0, 10),
    #    'zlim': (291.103905916, 325),
    #},
    (2, 4): {
        'xlim': (30, 40),
        'ylim': (0, 4),
    },
    (3, 12): [
        {
            'xlim': (-20, 10),
            'ylim': (0, 1.5),
        },
        {
            'xlim': (0, 10),
            'ylim': (0, 1.8),
        },
        {
            'xlim': (5, 40),
            'ylim': (0, 2.1),
        }
    ],
    (3, 14): {
        'xlim': (10, 40),
        'ylim': (0, 2.6),
    }
}


def load_records(index=None, data_dir=None, verbose=False):
    """Load the experiment records from Gothenburg April 2018. Notes on missing
    synchronizations and repeated trials are applied.

    Parameters:
    index: int or slice or index array or None, valid numpy array index to
           specify which records to load. A value of None will load all records.
           Defaults to None.
    data_dir: Specifies path to data directory containing bicycle and lidar log
              files. The lidar log files must be pre-processed as this function
              does not handle bag files.
    verbose: bool, print status messages while loading records
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

    if not isinstance(exp_index, collections.Iterable):
        exp_index = [exp_index]

    # determine which trials have specific bounding boxes
    trial_bbkeys = {}
    for i, j in TRIAL_BBMASK.keys():
        if i in exp_index:
            l = trial_bbkeys.get(i, [])
            l.append((i, j))
            trial_bbkeys[i] = l

    for i in exp_index:
        if verbose:
            print('creating record from {} and {}'.format(
                LIDAR_LOG_FILES[i], BICYCLE_LOG_FILES[i]))
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

        # recalculate event detection if required
        if i in trial_bbkeys:
            for key in trial_bbkeys[i]:
                rider_id, trial_id = key
                assert i == rider_id
                r.trials[trial_id]._detect_event(TRIAL_BBMASK[key])

        # append processed record
        exp_records.append(r)
        if verbose:
            print('created record from {} and {}'.format(
                LIDAR_LOG_FILES[i], BICYCLE_LOG_FILES[i]))

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

def instructed_eventtype(record_id, trial_id):
    """Return instructed event type for experiment trial.
    """
    # special case due to error during data collection
    if record_id == 5 and trial_id == 16:
        return trial2.EventType.Braking

    return trial2.EventType((((trial_id // 3) % 2) + (record_id % 2)) % 2)
