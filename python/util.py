#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def get_contiguous_numbers(x):
    from operator import itemgetter
    from itertools import groupby
    ranges = []
    for k, g in groupby(enumerate(x), lambda y: y[0] - y[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
    return ranges


def check_valid_record(rec):
    # check that rec dtype doesn't contain nested dtypes
    assert all(np.issubdtype(rec.dtype[i], np.number)
               for i in range(len(rec.dtype)))

    # time is the first field
    assert rec.dtype.names[0] == 'time'


def get_subplot_grid(rec):
    check_valid_record(rec)
    n = len(rec.dtype.names) - 1
    cols = 3 if (not n % 3) and (n > 6) else 2
    rows = int(np.ceil(n / cols))
    return rows, cols


def signal_unit(s):
    # return the SI unit for a signal name
    if s.startswith('accelerometer'):
        return 'm/s^2'
    elif s.startswith('gyroscope'):
        return 'rad/s'
    elif 'angle' in s:
        return 'rad'
    elif s.startswith('speed'):
        return 'm/s'
    else:
        raise ValueError('unit for signal {} is not defined'.format(s))


def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='same')


def outlier_index(x, y, N):
    diff = x - running_mean(x, N)
    return np.reshape(np.argwhere(np.abs(diff) > y), (-1,))

