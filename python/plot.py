#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
import scipy.stats
import matplotlib.lines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import filter as ff
import plot_braking as braking
import plot_steering as steering
import path
import util


def plot_timeseries(rec):
    util.check_valid_record(rec)

    names = rec.dtype.names
    t = rec[names[0]]
    signals = names[1:]
    colors = sns.color_palette('husl', len(signals))

    rows, cols = util.get_subplot_grid(rec)
    fig, axes = plt.subplots(rows, cols, sharex=True)
    for ax, signal, color in zip(axes.ravel(), signals, colors):
        ax.plot(t, rec[signal], label=signal, color=color)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(util.signal_unit(signal))
        ax.legend()

    return fig, axes


def plot_stft(rec, window_time_duration=1, subplot_grid=True):
    # window time duration: in seconds, larger value gives higher frequency
    # resolution
    util.check_valid_record(rec)

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
        rows, cols = util.get_subplot_grid(rec)
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


def load_records(rider_id=None, trial_id=None):
    if rider_id is None:
        rider_id = range(1, 17)
    if trial_id is None:
        trial_id = range(1, 5)
    recs = []
    for rid in rider_id:
        path = os.path.join(os.path.dirname(__file__),
                r'../data/etrike/experiment/rider{}/convbike/*.csv'.format(rid))
        filenames = glob.glob(path)
        for tid, f in enumerate(filenames, 1):
            if tid not in trial_id:
                continue
            try:
                r = record.load_file(f, cd['convbike'])
            except IndexError:
                continue
            recs.append((rid, tid, r))
    return recs


def make_stats(recs, dtype):
    stats = np.array([], dtype)
    for rid, tid, r in recs:
        try:
            if dtype == braking.metrics_dtype:
                metrics, _, _, _ = braking.get_metrics(r)
            elif dtype == steering.metrics_dtype:
                #if not (tid == 3 or tid == 4):
                if not tid == 4:
                    continue
                metrics = steering.get_metrics(r)
            # rider id and trial id aren't available within the record datatype
            # so we need to add them here
            metrics['rider id'] = rid
            metrics['trial id'] = tid
        except (TypeError, AssertionError):
        #except TypeError:
            continue
        stats = np.hstack((stats, metrics))
    return stats


if __name__ == '__main__':
    from matplotlib.backends.backend_pdf import PdfPages
    #pp = PdfPages('braking_plots.pdf')
    #pp = PdfPages('steering_plots.pdf')
    #pp = PdfPages('path_plots.pdf')

    #def save_fig(fig):
    #    fig.set_size_inches(12.76, 7.19)
    #    fig.tight_layout()
    #    pp.savefig(fig)

    import record
    import pickle
    with open('config.p', 'rb') as f:
        cd = pickle.load(f)

    recs = load_records()

    ## braking plots
    #stats = make_stats(recs, braking.metrics_dtype)

    #for rid in range(1, 17):
    #    fig, axes = braking.plot_rider_braking_events(recs, rid)
    #    save_fig(fig)
    #    fig, axes = braking.plot_rider_velocities(recs, rid)
    #    save_fig(fig)
    #fig, axes = braking.plot_histograms(stats)
    #save_fig(fig)

    ## steering plots
    #stats = make_stats(recs, steering.metrics_dtype)

    #for rid, tid, r in recs:
    #    if tid == 3 or tid == 4:
    #    #if tid == 4:
    #        fig, axes = plot_timeseries(r)
    #        fig.suptitle('rider {} trial {}'.format(rid, tid))
    #        save_fig(fig)

    #        k = 10
    #        try:
    #            fig, ax, k_freq = steering.plot_fft(r, k, 1.5)
    #        except AssertionError:
    #            print('kth highest frequency is greater than 1.5 Hz '
    #                  'for rider {} trial {}'.format(rid, tid))
    #            continue
    #        ax.set_title('steer angle fft for rider {} trial {}'.format(rid,
    #                                                                    tid))
    #        save_fig(fig)

    #        fig, ax = steering.plot_filtered(r)
    #        ax.set_title('filtered steer angle for rider {} trial {}'.format(
    #            rid, tid)) #        save_fig(fig)

    #fig, axes = steering.plot_histograms(stats)
    #save_fig(fig)
    #grids = steering.plot_bivariates(stats)
    #for g in grids:
    #    save_fig(g.fig)
    #fig, axes = steering.plot_swarms(stats)
    #save_fig(fig)

    ## trajectory plots
    for rid, tid, r in recs:
        if rid > 1:
            break
        if tid == 3 or tid == 4:
            soln, fig, axes, fig2, ax2 = path.get_trajectory(
                    r,
                    velocity_window_size=55,
                    yaw_rate_window_size=11,
                    plot=True,
                    trial_id=tid)
            fig.suptitle('filtered signals rider {} trial {}'.format(
                rid, tid))
            #save_fig(fig)

            fig2.suptitle('trajectory rider {} trial {}'.format(rid, tid))
            #save_fig(fig2)
            print('generated trajectory {} {}'.format(rid, tid))

    plt.show()
    #pp.close()
