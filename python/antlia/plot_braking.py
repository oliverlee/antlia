#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial
import scipy.stats
import matplotlib.lines
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt
import pandas as pd
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
    ('braking starttime', '<f8'),
    ('braking endtime', '<f8'),
    ('window size', '<i8'),
    ('braking range', '<i8', 2),
    ('lockup ranges', '<i8'),
    ('rider id', '<i8'),
    ('trial id', '<i8')
])

yfields = [('starting velocity', 'm/s'),
           ('braking duration', 's'),
           ('braking distance', 'm')]


def get_trial_braking_indices(accel, threshold=0.3, min_size=15):
    min_merge_accel = -0.15
    range_clumps = np.ma.extras._ezclump(accel > threshold)

    merged_clumps = []
    while range_clumps:
        if len(range_clumps) == 1:
            merged_clumps.append(range_clumps.pop())
        else:
            clump1 = range_clumps.pop(0)
            clump2 = range_clumps[0]
            if (((clump2.start - clump1.stop) < min_size) and
                np.all(accel[clump1.stop:clump2.start] > min_merge_accel)):
                range_clumps[0] = slice(clump1.start, clump2.stop)
            else:
                merged_clumps.append(clump1)
    merged_clumps = [(clump.start, clump.stop - 1) for clump in merged_clumps
                     if clump.stop - clump.start > min_size]
    if not merged_clumps:
        msg = 'Braking not detected. Use different parameters to specify '
        msg += 'braking conditions.'
        raise ValueError(msg)

    # find the 'largest' range by simplified integration
    largest = None
    for clump in merged_clumps:
        if largest is None:
            largest = clump
        elif sum(accel[slice(*clump)]) > sum(accel[slice(*largest)]):
            largest = clump

    return largest, merged_clumps


def get_metrics(trial, window_size=55, braking_threshold=0.3, min_size=15):
    """ window size in samples
        ws = 55 # window size of samples -> 0.44 seconds @ 125 Hz
    """
    t = trial['time']
    v = trial['speed']
    filtered_velocity = ff.moving_average(v,
                                          window_size, window_size/2)
    filtered_acceleration = ff.moving_average(trial['accelerometer x'],
                                              window_size, window_size/2)
    braking_range, _ = get_trial_braking_indices(
            filtered_acceleration, braking_threshold, min_size)
    b0, b1 = braking_range

    # determine if wheel lockup occurs
    # look at raw speed and filtered acceleration
    lockup_indices = np.where((v < 0.2) &
                              (filtered_acceleration > 2.5))[0]
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
    braking_starttime = t[b0]
    braking_endtime = t[b1]
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
                     braking_starttime,
                     braking_endtime,
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
        for clump in all_ranges:
            ax.axvspan(t[clump.start],
                       t[clump.stop],
                       color=colors[5], alpha=0.2)
        if largest_range is not None:
            ax.axvspan(t[largest_range.start],
                       t[largest_range.stop],
                       color=colors[5], alpha=0.4)
    return fig, axes


def plot_trial_braking_event(trial, ax=None, metrics_kw=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    colors = sns.color_palette('Paired', 10)
    vc = colors[1]
    ac = colors[3]

    try:
        if metrics_kw is None:
            metrics_kw = {}
        metrics, vf, af, lockup_ranges = get_metrics(trial, **metrics_kw)
    except TypeError:
        # skip empty input file
        return fig, ax

    t = trial['time']
    l0, l1 = metrics['braking range'][0]
    # fit to 10 second window
    tw = 10
    tb = t[l1] - t[l0]
    assert tb < tw
    i0 = 0
    i1 = -1

    # plot filtered signals
    ax.plot(t[i0:i1], vf[i0:i1], color=vc,
            label='velocity, gaussian weighted moving average')
    ax.plot(t[i0:i1], af[i0:i1], color=ac,
            label='acceleration, gaussian weighted moving average')
    ax.legend()
    ylim = ax.get_ylim()

    # plot unfiltered signals
    ax.plot(t[i0:i1], trial['speed'][i0:i1], color=vc, alpha=0.3)
    ax.plot(t[i0:i1], trial['accelerometer x'][i0:i1], color=ac, alpha=0.3)
    ax.axhline(0, color='black', linewidth=1,zorder=1)
    ax.set_ylim(ylim) # use ylim based on filtered data

    # plot braking section
    ax.axvspan(t[l0], t[l1], color=colors[5], alpha=0.3)
    # plot lockup sections in braking section
    for lr in lockup_ranges:
        if lr[1] < i0 or lr[0] > i1:
            # skip plotting of lockup ranges outside the event
            continue
        ax.axvspan(t[lr[0]], t[lr[1]], color=colors[7], alpha=0.5)

    # plot best fit line
    p = [metrics['linregress slope'], metrics['linregress intercept']]
    ax.plot(t[l0:l1], np.polyval(p, t[l0:l1]), color=colors[7])

    ax.set_ylabel('m/s, -m/s^2')
    ax.set_xlabel('time [s]')
    return fig, ax


def plot_rider_braking_events(recs, rider_id, **kwargs):
    fig, axes = plt.subplots(2, 2, **kwargs)
    axes = axes.ravel()

    colors = sns.color_palette('Paired', 10)
    for rid, tid, trial in recs:
        if rider_id != rid:
            continue

        plot_trial_braking_event(trial, axes[tid])
        ax.set_title('rider {} trial {}'.format(rid, tid))
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


def plot_bivariates(stats, show_hull=False):
    current_palette = sns.utils.get_color_cycle()
    n_colors = stats['rider id'].max() + 1

    if n_colors <= len(current_palette):
        colors = sns.color_palette(n_colors=n_colors)
    else:
        colors = sns.husl_palette(n_colors, l=.7)

    riders = np.unique(stats['rider id'])
    proxy_lines = []
    for rid in riders:
        c = colors[rid]
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
                     color=list(map(lambda x: colors[x], stats['rider id'])))
        g.ax_joint.legend(handles=proxy_lines, ncol=2, title=
                'pearson r = {:.2g}, p = {:.2g}'.format(
                    *scipy.stats.pearsonr(x, y)))
        g.set_axis_labels('slope [m/s^2]', '{} [{}]'.format(yf, unit))
        g.fig.suptitle('scatterplots of braking events')
        g.fig.set_size_inches(12.76, 7.19) # fix size for pdf save

        if show_hull:
            patches = []
            for rid in riders:
                index = stats['rider id'] == rid
                m = stats[index][['linregress slope', name]].copy()
                X = m.view(np.float64).reshape(m.shape[0], -1)

                hull = scipy.spatial.ConvexHull(X)
                polygon = matplotlib.patches.Polygon(X[hull.vertices, :],
                                                     closed=True,
                                                     zorder=1,
                                                     facecolor=colors[rid])
                patches.append(polygon)
            p = matplotlib.collections.PatchCollection(patches,
                                                       match_original=True,
                                                       alpha=0.05)
            g.ax_joint.add_collection(p)

        grids.append(g)
    return grids


def get_dataframe(stats):
    df = pd.DataFrame(stats[[
            'linregress slope',
            'linregress intercept',
            'linregress r-value',
            'linregress p-value',
            'linregress stderr',
            'starting velocity',
            'braking duration',
            'braking distance',
            'braking starttime',
            'braking endtime',
            'window size',
            #'braking range', pandas data frame data must be 1-dimensional
            'lockup ranges',
            'rider id',
            'trial id']])
    return df


def plot_swarms(stats, **kwargs):
    fig, axes = plt.subplots(4, 1, sharex=True, **kwargs)
    fig.suptitle('swarm plot of braking metrics per rider')
    axes = axes.ravel()
    yfields.append(('linregress slope', 'm/s^2'))

    df = get_dataframe(stats)
    for yf, ax in zip(yfields, axes):
        y = yf[0]
        sns.swarmplot(x='rider id', y=y, ax=ax, data=df, hue='rider id')
        ax.set_ylabel('{} [{}]'.format(yf[0], yf[1]))
        ax.legend().remove()
    ax.set_xlabel('rider id')
    return fig, axes
