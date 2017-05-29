#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import re
import sys
import numpy as np
import scipy.stats
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

braking_metrics_dtype = np.dtype([
    ('linregress slope', '<f8'),
    ('linregress intercept', '<f8'),
    ('linregress r-value', '<f8'),
    ('linregress p-value', '<f8'),
    ('linregress stderr', '<f8'),
    ('starting velocity', '<f8'),
    ('braking duration', '<f8'),
    ('braking distance', '<f8'),
    ('window size', '<i8'),
    ('braking range', '<i8', 2),
    ('lockup ranges', '<i8'),
    ('rider id', '<i8')
])

def get_braking_metrics(rec, window_size=55):
    """ window size in samples
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
    """
    t = rec['time']
    v = rec['speed']
    filtered_velocity = ff.moving_average(v,
                                          window_size, window_size/2)
    filtered_acceleration = ff.moving_average(rec['accelerometer x'],
                                              window_size, window_size/2)
    braking_range, _ = trial_braking_indices(filtered_acceleration)
    b0, b1 = braking_range

    # determine if wheel lockup occurs
    # look at raw speed and filtered acceleration
    lockup_indices = np.where((v < 0.2) &
                              (filtered_acceleration > 3))[0]
    lockup_ranges = get_contiguous_numbers(lockup_indices)

    # calculate braking indices by removing lockup ranges
    br = set(range(b0, b1))
    for l0, l1 in lockup_ranges:
        br -= set(range(l0, l1))
    braking_indices = list(br)

    # best-fit line metrics
    slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(
            t[braking_indices], v[braking_indices])
    # braking metrics
    start_velocity = filtered_velocity[b0]
    braking_duration = t[b1] - t[b0]
    # TODO what if best-fit line crosses zero?
    braking_distance = (np.polyval([slope, intercept], [t[b0], t[b1]]).mean() *
                        braking_duration)
    # TODO do filtering in another function and pass filtered signals to this
    # function to calculate metrics
    return (np.array([(slope,
                     intercept,
                     r_value,
                     p_value,
                     stderr,
                     start_velocity,
                     braking_duration,
                     braking_distance,
                     window_size,
                     braking_range,
                     len(lockup_ranges),
                     0)], dtype=braking_metrics_dtype),
            filtered_velocity,
            filtered_acceleration,
            lockup_ranges)


def plot_trial_braking_events(trial_dir, calibration_dict):
    pathname = os.path.join(trial_dir, '*.csv')
    filenames = glob.glob(pathname)

    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()
    recs = []
    stats = np.array([], dtype=braking_metrics_dtype)
    colors = sns.color_palette('Paired', 10)
    for i, (f, ax) in enumerate(zip(filenames, axes), 1):
        rider_id = re.search(r'rider([\d]+)/', f).group(1)
        try:
            r = record.load_file(f, calibration_dict)
            recs.append(r)
        except IndexError as e:
            print(e)
            continue
        t = r['time']
        try:
            metrics, vf, af, lockup_ranges = get_braking_metrics(r)
        except TypeError:
            # skip empty input file
            continue
        vc = colors[1]
        ac = colors[3]
        l0, l1 = metrics['braking range'][0]
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
        # plot braking section
        ax.axvspan(t[l0], t[l1], color=colors[5], alpha=0.3)
        # plot lockup sections
        for lr in lockup_ranges:
            ax.axvspan(t[lr[0]], t[lr[1]], color=colors[7], alpha=0.5)

        # plot best fit line
        p = [metrics['linregress slope'], metrics['linregress intercept']]
        ax.plot(t[l0:l1], np.polyval(p, t[l0:l1]), color=colors[7])

        # fill in rider id
        metrics['rider id'] = rider_id
        stats = np.hstack((stats, metrics))
    return fig, axes, recs, stats


if __name__ == '__main__':
    import record
    import pickle
    with open('config.p', 'rb') as f:
        cd = pickle.load(f)

    stats = np.array([], dtype=braking_metrics_dtype)
    rider_id = range(1, 17)
    if len(sys.argv) > 1:
        rider_id = sys.argv[1:]
    for rid in rider_id:
        #path = os.path.join(os.path.dirname(__file__),
        #        r'../data/etrike/experiment/rider{}/convbike/'.format(rid))
        #fig, axes, recs, st = plot_trial_braking_events(path, cd['convbike'])
        #fig.suptitle('rider {}'.format(rid))
        #stats = np.hstack((stats, st))
        path = os.path.join(os.path.dirname(__file__),
                r'../data/etrike/experiment/rider{}/convbike/*.csv'.format(rid))
        filenames = glob.glob(path)
        for f in filenames:
            try:
                r = record.load_file(f, cd['convbike'])
            except IndexError:
                continue
            try:
                metrics, _, _, _ = get_braking_metrics(r)
                metrics['rider id'] = rid # FIXME allow rider id to be passed as function input
            except TypeError:
                continue
            stats = np.hstack((stats, metrics))

    colors = sns.husl_palette(6, s=.8, l=.5)
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('histograms of braking events')
    axes = axes.ravel()
    field = [('slope of regression line [m/s^2]', 'linregress slope', None),
             ('square of correlation coefficient', 'linregress r-value', lambda x: x**2),
             ('standard error of estimated gradient', 'linregress stderr', None),
             ('starting velocity [m/s]', 'starting velocity', None),
             ('braking duration [s]', 'braking duration', None),
             ('braking distance [m]', 'braking distance', None),]
    for ax, f, c in zip(axes, field, colors):
        label, fieldname, func = f
        x = stats[fieldname]
        if func is not None:
            x = func(x)
        sns.distplot(x, ax=ax, color=c, label=label, kde=False)
        ax.legend()

    yfields = [('starting velocity', 'm/s'),
               ('braking duration', 'm'),
               ('braking distance', 's')]
    colors = sns.husl_palette(stats['rider id'].max() + 1, l=.7)
    riders = np.unique(stats['rider id'])
    proxy_lines = []
    for rid in riders:
        c = colors[rid]
        l = matplotlib.lines.Line2D([], [],
                linestyle='', marker='o', markerfacecolor=c,
                label='rider {}'.format(rid))
        proxy_lines.append(l)

    for yf in yfields:
        name, unit = yf
        x = stats['linregress slope']
        y = stats[name]
        g = sns.JointGrid(x=x, y=y)
        g.plot_marginals(sns.distplot, kde=False,
                         color=sns.xkcd_palette(['charcoal'])[0])
        g.plot_joint(plt.scatter,
                     color=list(map(lambda x: colors[x], stats['rider id'])))
        g.ax_joint.legend(handles=proxy_lines, ncol=2, title=
                'pearson r = {:.2g}, p = {:.2g}'.format(
                    *scipy.stats.pearsonr(x, y)))
        g.set_axis_labels('slope [m/s^2]', '{} [{}]'.format(yf, unit))
        g.fig.suptitle('scatterplots of braking events')

    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.suptitle('swarm plot of braking metrics per rider')
    axes = axes.ravel()
    yfields.append(('linregress slope', 'm/s^2'))
    for yf, ax in zip(yfields, axes):
        y = stats[yf[0]]
        x = stats['rider id']
        sns.swarmplot(x=x, y=y, ax=ax);
        ax.set_ylabel('{} [{}]'.format(yf[0], yf[1]))
    ax.set_xlabel('rider id')

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
