#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats
import matplotlib.lines
import matplotlib.pyplot as plt
import seaborn as sns

from antlia import filter as ff
from antlia import util

metrics_dtype = np.dtype([
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
    ('rider id', '<i8'),
    ('trial id', '<i8')
])

yfields = [('starting velocity', 'm/s'),
           ('braking duration', 'm'),
           ('braking distance', 's')]


def get_trial_braking_indices(accel, threshold=0.3, min_size=15):
    indices = np.where(accel > threshold)[0]
    ranges = util.get_contiguous_numbers(indices)

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


def get_metrics(rec, window_size=55):
    """ window size in samples
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
    """
    t = rec['time']
    v = rec['speed']
    filtered_velocity = ff.moving_average(v,
                                          window_size, window_size/2)
    filtered_acceleration = ff.moving_average(rec['accelerometer x'],
                                              window_size, window_size/2)
    braking_range, _ = get_trial_braking_indices(filtered_acceleration)
    b0, b1 = braking_range

    # determine if wheel lockup occurs
    # look at raw speed and filtered acceleration
    lockup_indices = np.where((v < 0.2) &
                              (filtered_acceleration > 3))[0]
    lockup_ranges = util.get_contiguous_numbers(lockup_indices)

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
                     0,
                     0)], dtype=metrics_dtype),
            filtered_velocity,
            filtered_acceleration,
            lockup_ranges)


def plot_rider_velocities(recs, rider_id, **kwargs):
    fig, axes = plt.subplots(2, 2, **kwargs)
    axes = axes.ravel()

    colors = sns.color_palette('Paired', 8)
    for rid, tid, r in recs:
        if rider_id != rid:
            continue
        t = r['time']
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
        vf = ff.moving_average(r['speed'], ws, ws/2)
        af = ff.moving_average(r['accelerometer x'], ws, ws/2)
        vc = colors[1]
        ac = colors[3]

        ax = axes[tid - 1]
        ax.plot(t, vf, label='velocity, gaussian weighted moving average',
                color=vc)
        ax.plot(t, af, label='acceleration, gaussian weighted moving average',
                color=ac)
        ax.legend()
        ylim = ax.get_ylim()
        ax.plot(t, r['speed'], color=vc, alpha=0.3)
        ax.plot(t, r['accelerometer x'], color=ac, alpha=0.3)
        ax.set_ylim(ylim)
        ax.set_title('rider {} trial {}'.format(rid, tid))
        ax.set_ylabel('m/s, -m/s^2')
        ax.set_xlabel('time [s]')
        ax.axhline(0, color=sns.xkcd_palette(['charcoal'])[0],
                   linewidth=1,zorder=1)

        largest_range, all_ranges = get_trial_braking_indices(af)
        for r0, r1 in all_ranges:
            ax.axvspan(t[r0], t[r1], color=colors[5], alpha=0.2)
        if largest_range is not None:
            ax.axvspan(t[largest_range[0]], t[largest_range[1]],
                       color=colors[5], alpha=0.4)
    return fig, axes

def plot_rider_braking_events(recs, rider_id, **kwargs):
    fig, axes = plt.subplots(2, 2, **kwargs)
    axes = axes.ravel()

    colors = sns.color_palette('Paired', 10)
    for rid, tid, r in recs:
        if rider_id != rid:
            continue
        t = r['time']
        try:
            metrics, vf, af, lockup_ranges = get_metrics(r)
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

        ax = axes[tid - 1]
        ax.plot(t[i0:i1], vf[i0:i1], color=vc,
                label='velocity, gaussian weighted moving average')
        ax.plot(t[i0:i1], af[i0:i1], color=ac,
                label='acceleration, gaussian weighted moving average')
        ax.legend()
        ylim = ax.get_ylim()
        ax.plot(t[i0:i1], r['speed'][i0:i1], color=vc, alpha=0.3)
        ax.plot(t[i0:i1], r['accelerometer x'][i0:i1], color=ac, alpha=0.3)
        ax.set_ylim(ylim)
        ax.set_title('rider {} trial {}'.format(rid, tid))
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
    return fig, axes


def plot_histograms(stats, **kwargs):
    colors = sns.husl_palette(6, s=.8, l=.5)
    fig, axes = plt.subplots(2, 3, **kwargs)
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
    return fig, axes


def plot_bivariates(stats):
    colors = sns.husl_palette(stats['rider id'].max() + 1, l=.7)
    riders = np.unique(stats['rider id'])
    proxy_lines = []
    for rid in riders:
        c = colors[rid - 1]
        l = matplotlib.lines.Line2D([], [],
                linestyle='', marker='o', markerfacecolor=c,
                label='rider {}'.format(rid))
        proxy_lines.append(l)

    grids = []
    for yf in yfields:
        name, unit = yf
        x = stats['linregress slope']
        y = stats[name]
        g = sns.JointGrid(x=x, y=y)
        g.plot_marginals(sns.distplot, kde=False,
                         color=sns.xkcd_palette(['charcoal'])[0])
        g.plot_joint(plt.scatter,
                     color=list(map(lambda x: colors[x - 1], stats['rider id'])))
        g.ax_joint.legend(handles=proxy_lines, ncol=2, title=
                'pearson r = {:.2g}, p = {:.2g}'.format(
                    *scipy.stats.pearsonr(x, y)))
        g.set_axis_labels('slope [m/s^2]', '{} [{}]'.format(yf, unit))
        g.fig.suptitle('scatterplots of braking events')
        g.fig.set_size_inches(12.76, 7.19) # fix size for pdf save
        grids.append(g)
    return grids


def plot_swarms(stats, **kwargs):
    fig, axes = plt.subplots(4, 1, sharex=True, **kwargs)
    fig.suptitle('swarm plot of braking metrics per rider')
    axes = axes.ravel()
    yfields.append(('linregress slope', 'm/s^2'))
    import pandas as pd
    df = pd.DataFrame(stats[[
    'linregress slope',
    'linregress intercept',
    'linregress r-value',
    'linregress p-value',
    'linregress stderr',
    'starting velocity',
    'braking duration',
    'braking distance',
    'window size',
    #'braking range', pandas data frame data must be 1-dimensional
    'lockup ranges',
    'rider id',
    'trial id',
        ]])
    for yf, ax in zip(yfields, axes):
        y = yf[0]
        sns.swarmplot(x='rider id', y=y, ax=ax, data=df, hue='rider id')
        ax.set_ylabel('{} [{}]'.format(yf[0], yf[1]))
        ax.legend().remove()
    ax.set_xlabel('rider id')
    return fig, axes
