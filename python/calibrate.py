#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import numpy as np

CALIBRATION_TYPES = ['convbike', 'ebike', 'etrike']
CALIBRATION_SPEEDS = np.array([0, 5, 10, 15, 20]) # kph
CALIBRATION_ANGLES = np.array([-60, -40, -20, 0, 20, 40, 60]) # degrees
CALIBRATION_STEER_SAMPLE_LENGTH = 120 # samples (1 sec @ 120 Hz)


def find_first(data, element):
    # find first element still not implemented in numpy
    i = np.argmax(data == element)
    if i == 0:
        assert data[i] == element
    return i


def find_last(data, element):
    return len(data) - 1 - find_first(data[::-1], element)


def load_file(filename):
    # line 2 missing time value in files:
    # data/calibration/ebike/speed/10kph.csv
    # data/calibration/etrike/steer_angle/3.csv
    return np.recfromcsv(filename, delimiter=',', invalid_raise=False)


def create_empty_config(calibration_types):
    config = {}
    for t in calibration_types:
        config[t] = {}
    return config


def get_calibration_config(data_path, calibration_types=None):
    # data_path is path to calibration data directory
    if calibration_types is None:
        ct = CALIBRATION_TYPES
    config = create_empty_config(ct)
    for t in ct:
        path = os.path.join(data_path, '{}/imu_static'.format(t))
        calibrate_imu(path, config[t])

        path = os.path.join(data_path, '{}/steer_angle'.format(t))
        calibrate_steer_angle(path, config[t])

        path = os.path.join(data_path, '{}/speed'.format(t))
        calibrate_speed(path, config[t])
    return config


def calibrate_imu(path, config):
    ''' path: path to directory containing csv file
        config: dictionary for a specific calibration type
                this dictionary is mutated
    '''
    # acceleration signals should show gravity in the +z direction
    pathname = os.path.join(path, '*.csv')
    filename = glob.glob(pathname)
    assert len(filename) == 1 # only one file for imu calibration
    r = load_file(filename[0])
    N = len(r) # number of samples
    names = r.dtype.names[3:]
    signals = ('accelerometer x',
               'accelerometer y',
               'accelerometer z',
               'gyroscope x',
               'gyroscope y',
               'gyroscope z')
    assert names == ('accx_g',
                     'accy_g',
                     'accz_g',
                     'gyrox_degs',
                     'gyroy_degs',
                     'gyroz_degs')
    for s, n in zip(signals, names):
        if n == 'accz_g':
            expected = 1 # in g
        else:
            expected = 0 # in g or deg/s
        if n.startswith('acc'):
            unit_factor = 9.81 # 1 g in m/s^2
        else:
            unit_factor = np.pi/180 # 1 deg/s in rad/s

        p0 = np.polyfit([expected] * N, r[n], 0)[0] # zero/1g offset
        if s == 'accz_g':
            expected_values = [p0 - 1, p0]
        else:
            expected_values = [p0, p0 + 1]
        p = np.polyfit(expected_values, [0, unit_factor], 1)
        config[s] = p


def calibrate_steer_angle(path, config):
    ''' path: path to directory containing csv files
        config: dictionary for a specific calibration type
                this dictionary is mutated
    '''
    angles = CALIBRATION_ANGLES
    filenames = glob.glob(path)
    if len(filenames) < len(angles):
        # etrike configuration is missing the most extreme angles
        angles = angles[1:-1]
    steer_rad = angles * np.pi/180 # deg to rad
    steer_dac = []
    for i, a in enumerate(angles, 1):
        # steer angle not constant in files:
        # data/calibration/ebike/steer_angle/6.csv
        # data/calibration/ebike/steer_angle/7.csv
        # use last N samples
        filename = os.path.join(path, '{}.csv'.format(i))
        r = load_file(filename)
        data = r.steerangle_lsb
        assert len(data) > CALIBRATION_STEER_SAMPLE_LENGTH
        steer_dac.append(np.mean(data[-CALIBRATION_STEER_SAMPLE_LENGTH:]))
    p = np.polyfit(steer_dac, steer_rad, 1) # use linear fit
    config['steer angle'] = p


def calibrate_speed(path, config):
    ''' path: path to directory containing csv files
        config: dictionary for a specific calibration type
                this dictionary is mutated
    '''
    speed_mps = CALIBRATION_SPEEDS * 1000 / 3600 # km/hr to m/s
    speed_dac = [0] # start with 0 -> 0
    for v in CALIBRATION_SPEEDS[1:]:
        # We skip acceleration to the constant velocity value by using the
        # first index that matches the signal median
        # We also search the data in reverse in the event of deceleration at
        # the end of a run
        filename = os.path.join(path, '{}kph.csv'.format(v))
        r = load_file(filename)
        data = r.speed_lsb
        med = np.median(data)
        first = find_first(data, med)
        last = find_last(data, med)
        assert last >= first
        speed_dac.append(np.mean(data[first:last]))
    p = np.polyfit(speed_dac, speed_mps, 1) # use linear fit
    config['speed'] = p


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        r'data/calibration')
    cfg1 = get_calibration_config(path)

    import pickle
    with open('config.p', 'wb') as f:
        pickle.dump(cfg1, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('config.p', 'rb') as f:
        cfg2 = pickle.load(f)
