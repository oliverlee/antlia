#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


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


def plot_timeseries(rec):
    check_valid_record(rec)

    names = rec.dtype.names
    t = rec[names[0]]
    signals = names[1:]
    colors = sns.color_palette('husl', len(signals))

    rows, cols = get_subplot_grid(rec)
    fig, axes = plt.subplots(rows, cols, sharex=True)
    for ax, signal, color in zip(axes.ravel(), signals, colors):
        ax.plot(t, rec[signal], label=signal, color=color)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(signal_unit(signal))
        ax.legend()

    return fig, axes


def fft(x, sample_period, window_type=None):
    if window_type is None:
        window_type = scipy.signal.hamming
    n = len(x)
    windowed_x = np.multiply(x, window_type(n))

    # only use first half of fft since real signals are mirrored about nyquist
    # frequency
    xf = 2/n * np.abs(scipy.fftpack.fft(windowed_x)[:n//2])
    freq = np.linspace(0, 1/(2*sample_period), n/2)
    return freq, xf


def rolling_fft(x, sample_period,
                window_start_indices, window_length, window_type=None):
    X = []
    for i in window_start_indices:
        freq, xf = fft(x[i:i + window_length], sample_period, window_type)
        X.append(xf)
    return freq, window_start_indices, X


def plot_stft(rec, window_time_duration=1, subplot_grid=True):
    # window time duration: in seconds, larger value gives higher frequency
    # resolution
    check_valid_record(rec)

    names = rec.dtype.names
    t = rec.time
    signals = names[1:]
    colors = sns.color_palette('husl', len(signals))

    sample_period = np.diff(t).mean()
    window_length = int(window_time_duration/sample_period)
    window_start_indices = range(0,
                                 len(t)//window_length * window_length,
                                 window_length)

    window_start_string = 'range(0, t[-1]//N*N, N), N = {} sec'.format(
            window_time_duration)
    figure_title = 'STFT, {} sec time window at times {}'.format(
            window_time_duration, window_start_string)

    if subplot_grid:
        rows, cols = get_subplot_grid(rec)
        fig = plt.figure()
    else:
        fig = [plt.figure() for _ in signals]
    axes = []

    for i, (signal, color) in enumerate(zip(signals, colors)):
        if subplot_grid:
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            fig.suptitle(figure_title)
        else:
            ax = fig[i].add_subplot(1, 1, 1, projection='3d')
            fig[i].suptitle(figure_title)
        start_times = t[window_start_indices]
        frequencies, _, amplitudes = rolling_fft(r[signal], sample_period,
                                                 window_start_indices,
                                                 window_length)
        X, Y = np.meshgrid(frequencies, start_times)
        Z = np.reshape(amplitudes, X.shape)
        ax.plot_surface(X, Y, Z,
                        rcount=len(frequencies), ccount=len(start_times),
                        color=color)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('time [s]')
        proxy = matplotlib.lines.Line2D([], [], color=color)
        ax.legend([proxy], [signal])
        axes.append(ax)
    return fig, axes



if __name__ == '__main__':
    import record
    import pickle
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import glob
        pathname = os.path.join(os.path.dirname(__file__),
                r'../data/etrike/experiment/rider3/convbike/*.csv')
        filenames = glob.glob(pathname)
        path = filenames[3]
    with open('config.p', 'rb') as f:
        cd = pickle.load(f)
    r2 = record.load_file(path, cd['convbike'])

    # fig, axes = plot_timeseries(r)
    # fig.suptitle(path)

    # prepend_path = lambda f, p: f.suptitle(
    #         '{}\n{}'.format(path, f._suptitle.get_text()))
    # fig2, axes2 = plot_stft(r, subplot_grid=True)
    # try:
    #     for f in fig2:
    #         prepend_path(f, path)
    # except TypeError:
    #     prepend_path(fig2, path)

    path = os.path.join(os.path.dirname(__file__),
                        '../data/20160107-113037_sensor_data.h5')
    r = record.load_file(path)

    # get slice of data since it is HUGE
    t = r.time
    i0 = np.argmax(t >= 4700)
    i1 = np.argmax(t >= 4800)
    rr = r[i0:i1]
    fig, axes = plot_timeseries(rr)
    fig.suptitle(path)

    fig2, axes2 = plot_stft(rr, subplot_grid=True)

    plt.show()
