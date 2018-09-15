#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import os
import pickle
import numpy as np

from antlia import dtype
from antlia import kalman
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
## time for trial (1, 2) determined manually

TRIAL_BBMASK = {
    (1, 2): {
        'xlim': (-5, -2.5),
        'ylim': (0, 10),
        'zlim': (291.5, 325),
    },
    (1, 10): {
        'xlim': (-20, 50),
        'ylim': (0, 10),
        'zlim': (0, 1100),
    },
    (2, 4): {
        'xlim': (30, 40),
        'ylim': (0, 4),
    },
    (2, 13): [
        {
            'xlim': (4, 4.4),
            'ylim': (3.2, 3.3),
        },
        {
            'xlim': (3.85, 4),
            'ylim': (2.99, 3.07),
        },
    ],
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
    },
    (10, 0): [
        {
            'xlim': (-20, 50),
            'ylim': (0, 5),
            'zlim': (0, 60)
        },
        {
            'xlim': (4, 7),
            'ylim': (1, 3),
        },
        {
            'xlim': (3, 5),
            'ylim': (1, 2.4),
        },
        {
            'xlim': (6, 8),
            'ylim': (3.25, 3.5),
        },
        {
            'xlim': (4.15, 4.25),
            'ylim': (3.25, 3.35),
        },
        {
            'xlim': (0, 10),
            'ylim': (3, 4),
            'zlim': (97, 120),
        }
    ],
    (10, 1): {
        'xlim': (5.2, 6.2),
        'ylim': (2.5, 3.0),
    },
    (10, 2): [
        {
            'xlim': (-20, -10),
            'ylim': (0, 5),
        },
        {
            'xlim': (3, 5),
            'ylim': (1.3, 2.5),
        }
    ],
    (10, 3): [
        {
            'xlim': (5.65, 5.85),
            'ylim': (2.76, 2.88),
        },
        {
            'xlim': (5.25, 5.60),
            'ylim': (2.55, 2.75),
        }
    ],
    (10, 4): [
        {
            'xlim': (3.3, 5.3),
            'ylim': (1.6, 2.5),
        },
        {
            'xlim': (5, 6),
            'ylim': (2.5, 2.9),
        }
    ],
    (14, 1): [
        {
            'xlim': (-20, -10),
            'ylim': (0, 5),
        },
        {
            'xlim': (0, 60),
            'ylim': (0, 5),
            'zlim': (100, 165),
        },
        {
            'xlim': (-10, 10),
            'ylim': (0, 2),
            'zlim': (165, 175),
        }
    ],
    (14, 6): [
        {
            'xlim': (-20, -10),
            'ylim': (0, 5),
        },
        {
            'xlim': (-10, -10),
            'ylim': (0, 540),
        },
        {
            'xlim': (0, 50),
            'ylim': (0, 10),
            'zlim': (0, 535),
        }
    ],
    (16, 2): [
        {
            'xlim': (4.1, 4.3),
            'ylim': (3.25, 3.33),
        },
        {
            'xlim': (-10, -5),
            'ylim': (0, 5),
        },
    ],
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
        r._calculate_trials2(instructed_record_eventtypes(i),
                             missing_sync=MISSING_SYNC[i],
                             trial_mask=TRIAL_MASK[i])

        # recalculate event detection if required
        if i in trial_bbkeys:
            for key in trial_bbkeys[i]:
                rider_id, trial_id = key
                assert i == rider_id
                r.trials[trial_id]._detect_event(instructed_eventtype(*key),
                                                 TRIAL_BBMASK[key])

        # append processed record
        exp_records.append(r)
        if verbose:
            print('created record from {} and {}'.format(
                LIDAR_LOG_FILES[i], BICYCLE_LOG_FILES[i]))

    return exp_records


def _estimate_state(records, record_ids=None):
    # generate Kalman matrices
    f, h, F, H = kalman.generate_fhFH(constant_velocity=True,
                                      wheelbase=0.6) #TODO verify

    T = 1/125 # bicycle sample rate
    q0 = 1 # weight factor for translation-related states
    q1 = 0.01 # weight factor for rotation-related states

    # process noise covariance matrix
    Q = 1*np.diag([
        q0*T**3/6, # [m] x-position
        q0*T**3/6, # [m] y-position
        q1*T**2/2, # [rad/s] yaw angle
        q0*T**2/2, # [m/s] velocity
        q1*T, # [rad/s] yaw rate
        q0*T/10, # [m/s^2] acceleration
    ])

    # initial error covariance matrix
    P0 = np.diag([
        0.1,
        0.1,
        0.01,
        1,
        0.1,
        0.2
    ])

    if not isinstance(records, collections.Iterable):
        records = [records]

    if record_ids is None:
        record_ids = range(len(records))

    for i, r in zip(record_ids, records):
        R = kalman.generate_R(r)

        for j, tr in enumerate(r.trials):
            event = tr.event

            # create measurement array from event data
            z = kalman.generate_measurement(event)

            # create initial state estimate
            x0 = kalman.initial_state_estimate(z)
            ## replace initial x-position estimate with max from data
            #x0[0] = event.x.max()
            # replace trajectory-derived velocity estimate with instructed speed
            x0[3] = instructed_speed(i, j)

            ## check position estimate is reasonable
            #assert x0[0] > 15, 'initial x: {:0.3f}'.format(x0[0])
            #assert x0[1] > 2.0 and x0[1] < 3.5, 'initial y: {:0.3f}'.format(x0[1])

            kf = kalman.Kalman(F, H, Q, R, f, h)
            result = kf.estimate(x0, P0, z)
            smoothed_result = kf.smooth_estimate(result)

            event.kalman_result = result
            event.kalman_smoothed_result = smoothed_result


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
        return trial2.EventType.Overtaking

    return trial2.EventType((((trial_id // 3) % 2) + (record_id % 2)) % 2)

def instructed_record_eventtypes(record_id):
    """Return array of instructed event types for experiment record.
    """
    types = [0, 0, 0,
             1, 1, 1,
             0, 0, 0,
             1, 1, 1,
             0, 0, 0,
             1, 1, 1]
    types = np.array([trial2.EventType(t) for t in types])

    if (record_id % 2) == 1:
        types = np.roll(types, 3)

    # special case due to error during data collection
    if record_id == 5:
        types[16] = trial2.EventType.Overtaking

    return types
