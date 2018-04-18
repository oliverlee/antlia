# -*- coding: utf-8 -*-
import gzip
import pickle
import numpy as np

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


def load_converted_record(filename):
    with gzip.open(filename, 'rb') as f:
        try:
            # specify encoding for Python 3
            record = pickle.load(f, encoding='bytes')
        except:
            record = pickle.load(f)
        finally:
            return record.view(np.recarray)
