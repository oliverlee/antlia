#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate
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
                       linewidth=1, zorder=1)
        ax.set_xlabel('time [s]')

    def func(y, t):
        xp, yp, yaw = y
        i = np.argmax(r['time'] >= t)
        v = vf[i]
        # xdot = v*cos(yaw)
        # ydot = v*sin(yaw)
        dydt = [v*np.cos(yaw), v*np.sin(yaw), yf[i]]
        return dydt

    soln = scipy.integrate.odeint(func, [0, 0, 0], r['time'])

    if plot:
        fig2, axes2 = plt.subplots(2, 1)
        x = soln[:, 0]
        y = soln[:, 1]
        yaw = soln[:, 2]
        axes2[0].plot(t, yaw, color=colors[7], label='yaw angle')
        axes2[0].axhline(0, color=sns.xkcd_palette(['charcoal'])[0],
                         linewidth=1, zorder=1)
        axes2[0].legend()

        axes2[1].plot(x, y, color=colors[0], label='trajectory')
        axes2[1].set(aspect='equal')
        ylim = axes2[1].get_ylim()
        axes2[1].set_ylim([ylim[0] - 5, ylim[1] + 5])
        axes2[1].plot([0, 128], [0, 0], color=sns.xkcd_palette(['charcoal'])[0],
                      label='trial length', linewidth=2, zorder=1)
        axes2[1].legend()

        return soln, fig, axes, fig2, axes2
    return soln
