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


def reduce_runs(signal, tol=1e-3):
    """Return a slice of the signal array with the runs reduced to 2 subsequent
    elements.

    Parameters:
    signal: array_like, data to reduce
    tol: float, tolerance used for element equality

    Returns:
    reduced_signal: ndarray.view, 'signal' slice with reduced runs
    reduced_indices: array_like, indices of reduced runs

    Notes: Must be able to compare elements.

    Example:
    > s = np.array([0, 0, 0, 1, 1, 2, 3, 0, 0])
    > reduce_signal(s)
    [0, 0, 1, 2, 3, 0, 0] [0, 2, 3, 4, 5, 6, 7, 8]
    """
    signal = np.asarray(signal)
    edge = np.where(np.abs(np.diff(signal)) > tol)[0] + 1

    # Include first element.
    index = [0]

    for i in edge:
        skip = i - index[-1]
        assert skip > 0

        # If we skip more than one index,
        # include previous index so edges are preserved.
        if skip > 1:
            index.append(i - 1)
        index.append(i)

    # Include last element if the signal ends in a run.
    n = len(signal)
    if index[-1] != n - 1:
        index.append(n - 1)

    return signal[index], index


def debounce(x, decay=100):
    """Debounce a digital signal.

    Parameters:
    x: array_like
    decay: int, size of edge filter used when debouncing an edge

    Example:
    > x = np.r_[0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
    > debounce(x).astype(int)
    [0 0 1 1 1 1 1 1 0 0]
    """
    edges = np.zeros(x.shape)
    edges[1:] = np.diff(x.astype(int))

    n = edges.shape[0]
    for i in range(n):
        if edges[i] == -1:
            for j in range(min(decay, n - i)):
                if edges[i + j] == 1:
                    edges[i] = 0
                    edges[i + j] = 0
                    break

    return np.cumsum(edges) + x[0]


if __name__ == '__main__':
    x = np.r_[0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
    print(x)
    print(debounce(x).astype(int))

    x2 = np.r_[1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
    print(x2)
    print(debounce(x2).astype(int))
