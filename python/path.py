#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns

import filter as ff
import util


def get_trajectory(r, velocity_window_size, yaw_rate_window_size, plot=False):
    vf = ff.moving_average(r['speed'],
                           velocity_window_size,
                           velocity_window_size/2)
    yf = ff.moving_average(r['gyroscope z'],
                           yaw_rate_window_size,
                           yaw_rate_window_size/2)

    if plot:
        colors = sns.color_palette('husl', 8)
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes = axes.ravel()

        t = r['time']
        for ax, c, sig, sigf, ws, l in zip(axes,
                               [colors[1], colors[7]],
                               [r['speed'], r['gyroscope z']],
                               [vf, yf],
                               [velocity_window_size, yaw_rate_window_size],
                               [['velocity', '[m/s]'],
                                ['yaw rate', '[rad/s]']]):
            label = '{}, gaussian weighted moving average, {} samples'.format(
                    l[0], ws)
            ax.plot(t, sigf, color=c, label=label)
            ylim = ax.get_ylim()
            ax.plot(t, sig, color=c, label=(l[0] + ', measured'), alpha=0.3)
            ax.set_ylim(ylim)
            ax.legend()
            ax.set_ylabel(l[-1])
            ax.axhline(0, color=sns.xkcd_palette(['charcoal'])[0],
                       linewidth=1,zorder=1)
        ax.set_xlabel('time [s]')
        return fig, ax
