#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import sys
import numpy as np
import matplotlib.lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import filter as ff


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
        frequencies, _, amplitudes = ff.rolling_fft(rec[signal],
                                                    sample_period,
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


def get_contiguous_numbers(x):
    from operator import itemgetter
    from itertools import groupby
    ranges = []
    for k, g in groupby(enumerate(x), lambda y: y[0] - y[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
    return ranges


def trial_braking_indices(accel, threshold=0.3, min_size=15):
    indices = np.where(accel > threshold)[0]
    ranges = get_contiguous_numbers(indices)

    # exclude ranges smaller than min_size
    ranges = [(r0, r1) for (r0, r1) in ranges if (r1 - r0) > min_size]

    # merge ranges that are separated by min_size if acceleration sign does not
    # change
    merged = []
    while ranges:
        if len(ranges) == 1:
            merged.append(ranges.pop(0))
        else:
            ra = ranges.pop(0)
            rb = ranges[0]
            if (((rb[0] - ra[1]) < min_size) and
                np.all(np.sign(accel[ra[1]:rb[0]]) == 1)):
                ranges[0] = (ra[0], rb[1])
            else:
                merged.append(ra)

    # find the 'largest' range by simplified integration
    largest = None
    for r0, r1 in merged:
        if largest is None:
            largest = (r0, r1)
        elif sum(accel[r0:r1]) > sum(accel[largest[0]:largest[1]]):
            largest = (r0, r1)
    return largest, merged


def plot_trial_velocities(trial_dir, calibration_dict):
    pathname = os.path.join(trial_dir, '*.csv')
    filenames = glob.glob(pathname)

    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()
    recs = []
    colors = sns.color_palette('Paired', 8)
    for i, (f, ax) in enumerate(zip(filenames, axes), 1):
        try:
            r = record.load_file(f, calibration_dict)
            recs.append(r)
        except IndexError as e:
            print(e)
            continue
        t = r['time']
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
        vf = ff.moving_average(r['speed'], ws, ws/2)
        af = ff.moving_average(r['accelerometer x'], ws, ws/2)
        vc = colors[1]
        ac = colors[3]
        ax.plot(t, vf, label='velocity, gaussian weighted moving average',
                color=vc)
        ax.plot(t, af, label='acceleration, gaussian weighted moving average',
                color=ac)
        ax.legend()
        ylim = ax.get_ylim()
        ax.plot(t, r['speed'], color=vc, alpha=0.3)
        ax.plot(t, r['accelerometer x'], color=ac, alpha=0.3)
        ax.set_ylim(ylim)
        ax.set_title('trial {}: {}'.format(i, os.path.basename(f)))
        ax.set_ylabel('m/s, -m/s^2')
        ax.set_xlabel('time [s]')
        ax.axhline(0, color=sns.xkcd_palette(['charcoal'])[0],
                   linewidth=1,zorder=1)

        largest_range, all_ranges = trial_braking_indices(af)
        for r0, r1 in all_ranges:
            ax.axvspan(t[r0], t[r1], color=colors[5], alpha=0.2)
        if largest_range is not None:
            ax.axvspan(t[largest_range[0]], t[largest_range[1]],
                       color=colors[5], alpha=0.4)
    return fig, axes, recs


def plot_trial_braking_events(trial_dir, calibration_dict):
    pathname = os.path.join(trial_dir, '*.csv')
    filenames = glob.glob(pathname)

    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()
    recs = []
    colors = sns.color_palette('Paired', 10)
    for i, (f, ax) in enumerate(zip(filenames, axes), 1):
        try:
            r = record.load_file(f, calibration_dict)
            recs.append(r)
        except IndexError as e:
            print(e)
            continue
        t = r['time']
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
        vf = ff.moving_average(r['speed'], ws, ws/2)
        af = ff.moving_average(r['accelerometer x'], ws, ws/2)
        vc = colors[1]
        ac = colors[3]
        largest_range, all_ranges = trial_braking_indices(af)
        if largest_range is None:
            continue
        else:
            l0, l1 = largest_range
            # fit to 10 second window
            tw = 10
            tb = t[l1] - t[l0]
            assert tb < tw
            # try to center
            i0 = int((l1 + l0)/2 - (l1 - l0)/2 * tw/tb)
            i1 = int((l1 + l0)/2 + (l1 - l0)/2 * tw/tb)
            # shift to left if the braking event can't be centered
            if i1 > len(r):
                n = i1 - len(r) + 1
                i0 -= n
                i1 -= n

        ax.plot(t[i0:i1], vf[i0:i1], color=vc,
                label='velocity, gaussian weighted moving average')
        ax.plot(t[i0:i1], af[i0:i1], color=ac,
                label='acceleration, gaussian weighted moving average')
        ax.legend()
        ylim = ax.get_ylim()
        ax.plot(t[i0:i1], r['speed'][i0:i1], color=vc, alpha=0.3)
        ax.plot(t[i0:i1], r['accelerometer x'][i0:i1], color=ac, alpha=0.3)
        ax.set_ylim(ylim)
        ax.set_title('trial {}: {}'.format(i, os.path.basename(f)))
        ax.set_ylabel('m/s, -m/s^2')
        ax.set_xlabel('time [s]')
        ax.axhline(0, color=sns.xkcd_palette(['charcoal'])[0],
                   linewidth=1,zorder=1)
        ax.axvspan(t[l0], t[l1], color=colors[5], alpha=0.3)

        # calculate and plot velocity fit
        # determine if rear wheel is locking up, in this case, measurement is
        # only of the rear wheel and not the bicycle and rider
        lockup_indices = np.where((r['speed'] < 0.2) & (af > 3))[0]
        lockup_ranges = get_contiguous_numbers(lockup_indices)
        poly_range = set(range(l0, l1))
        for lr0, lr1 in lockup_ranges:
            ax.axvspan(t[lr0], t[lr1], color=colors[7], alpha=0.5)
            poly_range -= set(range(lr0, lr1))

        poly_indices = list(poly_range)
        p = np.polyfit(t[poly_indices], r['speed'][poly_indices], 1)
        ax.plot(t[l0:l1], np.polyval(p, t[l0:l1]), color=colors[7])
    return fig, axes, recs


if __name__ == '__main__':
    import record
    import pickle
    with open('config.p', 'rb') as f:
        cd = pickle.load(f)

    rider_id = range(1, 17)
    if len(sys.argv) > 1:
        rider_id = sys.argv[1:]
    for rid in rider_id:
        path = os.path.join(os.path.dirname(__file__),
                r'../data/etrike/experiment/rider{}/convbike/'.format(rid))
        fig, axes, recs = plot_trial_braking_events(path, cd['convbike'])
        fig.suptitle('rider {}'.format(rid))

    #path = os.path.join(os.path.dirname(__file__),
    #                    '../data/20160107-113037_sensor_data.h5')
    #r = record.load_file(path)

    ## get slice of data since it is HUGE
    #t = r.time
    #i0 = np.argmax(t >= 4700)
    #i1 = np.argmax(t >= 4800)
    #rr = r[i0:i1]
    #fig, axes = plot_timeseries(rr)
    #fig.suptitle(path)

    plt.show()
