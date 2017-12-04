# -*- coding: utf-8 -*-
import glob
import os
import pickle
import numpy as np

from record import load_file


LIDAR_NUM_ANGLES = 1521
LIDAR_FOV_DEG = 190
LIDAR_SAMPLE_RATE = 20
LIDAR_ANGLES = np.linspace( # in radians
    (90 - LIDAR_FOV_DEG/2)*np.pi/180,
    (90 + LIDAR_FOV_DEG/2)*np.pi/180,
    LIDAR_NUM_ANGLES
)

"""LIDAR datatype format is:
    (
        timestamp (long),
        flag (bool saved as int),
        accelerometer[3] (double),
        gps[3] (double),
        distance[LIDAR_NUM_ANGLES] (long),
    )

    'int' and 'long' are the same size on the raspberry pi (32 bits).
"""
LIDAR_RECORD_DTYPE = np.dtype(','.join(
    ['i4'] + ['i4'] + 3*['f8'] + 3*['f8'] + LIDAR_NUM_ANGLES*['i4']
    )
)
LIDAR_CONVERTED_DTYPE = np.dtype([
    ('time', 'f8'),
    ('sync', 'f8'),
    ('accelerometer x', 'f8'), # body x-axis may not be aligned with inertial!
    ('accelerometer y', 'f8'), # body y-axis may not be aligned with inertial!
    ('accelerometer z', 'f8'), # body z-axis may not be aligned with inertial!
    ('gps', 'f8', (3,)),
    ('distance', 'f8', (LIDAR_NUM_ANGLES,)),
])

__record_path = os.path.join(os.path.dirname(__file__),
                             r'../data/lidar/')

def __get_record_files(extension):
    record_path = os.path.join(os.path.dirname(__file__),
                               r'../data/lidar/')
    return sorted(glob.glob('{}*{}'.format(record_path, extension)))


LIDAR_RECORD_FILES =  __get_record_files('.bin')
BICYCLE_RECORD_FILES = __get_record_files('.csv')


def get_lidar_records(convert_dtype=True):
    """Returns a list of LIDAR records as numpy structured array.
        convert_dtype[True]: Datype of LIDAR records.
            True    - dtype = LIDAR_CONVERTED_DTYPE
            False   - dtype = LIDAR_RECORD_DTYPE
    """
    float_dtype = np.dtype(','.join(len(LIDAR_RECORD_DTYPE) * ['f8']))

    records = []
    for filename in LIDAR_RECORD_FILES:
        with open(filename, 'rb') as f:
            x = np.fromfile(f, LIDAR_RECORD_DTYPE)
            if convert_dtype:
                # copy and convert data to new dtype
                y = x.astype(float_dtype)
                # create view which shares the same underlying memory
                x = y.view(LIDAR_CONVERTED_DTYPE).view(np.recarray)
                # convert timestamp to time and start at zero
                x.time /= 1000
                x.time -= x.time[0]
                # flip sync value to be active high
                x.sync = np.invert(x.sync.astype(bool)).astype(x.sync.dtype)
                # TODO: convert accelerometer
                # TODO: convert gps
                # convert distance from millimeters to meters
                x.distance /= 1000
            records.append(x)
    return records


def get_bicycle_records():
    # load calibration data
    calibration_path = os.path.join(os.path.dirname(__file__),
                                    r'config.p')
    with open(calibration_path, 'rb') as f:
        calibration = pickle.load(f)

    records = []
    for filename in BICYCLE_RECORD_FILES:
        r = load_file(filename, calibration['convbike'])
        records.append(r)
    return records
