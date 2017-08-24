#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate
#import scipy.signal
import matplotlib.patches
import matplotlib.pyplot as plt
import seaborn as sns

import filter as ff
import util


def plot_velocity(r, velocity_window_size):
    charcoal_color = sns.xkcd_palette(['charcoal'])[0]

    t = r['time']
    v = r['speed']
    dt = np.diff(t).mean()

    vf = ff.moving_average(v,
                           velocity_window_size,
                           velocity_window_size/3)

    #fs = 1/dt # sample frequency
    #f0 = 31.25 # notch frequency
    #Q = 1 # quality factor
    #w0 = f0/(fs/2) # normalized frequency
    #b, a = scipy.signal.iirnotch(w0, Q)
    #vf2 = scipy.signal.filtfilt(b, a, v)

    colors = sns.color_palette('husl', 8)
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(t, v, color=colors[0], alpha=0.3,
            label='velocity, measured')
    ax.plot(t, vf, color=colors[1],
            label='velocity, gaussian weighted moving average, {} samples'.format(
                velocity_window_size))
    #ax.plot(t, vf2, color=colors[3],
    #        label='velocity, notch filter, f0: {} Hz, Q: {}'.format(
    #            f0, Q))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('velocity [m/s]')
    ax.axhline(0, color=charcoal_color, linewidth=1, zorder=1)
    ax.legend()

    freq, xf = ff.fft(v, dt) # uses hamming window
    freq, xf1 = ff.fft(vf, dt) # uses hamming window
    #freq, xf2 = ff.fft(vf2, dt) # uses hamming window

    ax = axes[1]
    #markerline, stemline, baseline = ax.stem(freq, xf, markerfmt=' ')
    #plt.setp(stemline, 'color', colors[1], 'alpha', 0.3)
    ax.plot(freq, xf, color=colors[0], alpha=0.3)
    ax.plot(freq, xf1, color=colors[1])
    #ax.plot(freq, xf2, color=colors[3])
    ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('amplitude')


def get_trajectory(r, velocity_window_size, yaw_rate_window_size,
                  plot=False, trial_id=None):
    charcoal_color = sns.xkcd_palette(['charcoal'])[0]
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
            ax.axhline(0, color=charcoal_color,
                       linewidth=1, zorder=1)
        ax.set_xlabel('time [s]')

    def f(y, t):
        i = np.argmax(r['time'] >= t)
        return yf[i]
    yaw_angle = scipy.integrate.odeint(f, 0, r['time'])

    def func(y, t):
        xp, yp, yaw = y
        i = np.argmax(r['time'] >= t)
        v = vf[i]
        # xdot = v*cos(yaw)
        # ydot = v*sin(yaw)
        dydt = [v*np.cos(yaw), v*np.sin(yaw), yf[i]]
        return dydt

    soln = scipy.integrate.odeint(func, [0, 0, -yaw_angle.mean()], r['time'])

    if plot:
        fig2, axes2 = plt.subplots(2, 1)
        x = soln[:, 0]
        y = soln[:, 1]
        yaw = soln[:, 2]
        ax = axes2[0]
        ax.plot(t, yaw, color=colors[7], label='yaw angle')
        ax.axhline(0, color=charcoal_color, linewidth=1, zorder=1)
        ax.legend()

        ax = axes2[1]
        ax.plot(x, y, color=colors[0], label='trajectory')
        ax.set(aspect='equal')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0] - 5, ylim[1] + 5])
        ax.plot([0, 128], [0, 0], color=charcoal_color,
                label='trial path', linewidth=2, zorder=1)
        if trial_id == 3:
            num_cones = 2
            start_x = 60
            step_x = 5
        elif trial_id == 4:
            num_cones = 4
            start_x = 120 - 10
            step_x = -3

        cones = []
        circ_radius = 0.5
        for i in range(num_cones):
            circle = matplotlib.patches.Circle(
                    (start_x + i*step_x, 0),
                    circ_radius,
                    color=colors[1],
                    label='cones')
            cones.append(circle)
            ax.add_artist(circle)

        if cones:
            handles, _ = ax.get_legend_handles_labels()
            handles += cones[0:1]
            ax.legend(handles=handles, loc='center left')
        else:
            ax.legend(loc='center left')

        return soln, fig, axes, fig2, axes2
    return soln
