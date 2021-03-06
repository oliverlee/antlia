#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate
import scipy.signal
import matplotlib.patches
import matplotlib.pyplot as plt
import seaborn as sns

from antlia import filter as ff
from antlia import kalman
from antlia import util
from antlia.madgwick_py.madgwickahrs import MadgwickAHRS


def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='same')


def outlier_index(x, y, N):
    diff = x - running_mean(x, N)
    return np.reshape(np.argwhere(np.abs(diff) > y), (-1,))


    from antlia import kalman
def plot_vel_acc(r, velocity_window_size):
    t = r['time']
    v = r['speed']
    a = r['accelerometer x']
    dt = np.diff(t).mean()

    v_mov = ff.moving_average(v, velocity_window_size, velocity_window_size/3)

    u = np.reshape(a, (-1, 1, 1))
    z = np.reshape(v, (-1, 1, 1))
    oi = [x for x in util.outlier_index(v, 2*v.std(), 128) if x > 128]
    v_kal = kalman.kalman_velocity(dt, v, u, z, q=5, missed_measurement=oi)

    colors = sns.color_palette('Paired', 10)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(t, v, color=colors[0], alpha=0.5,
            label='velocity, measured')
    ax.plot(t[oi], v[oi], color=colors[0], linestyle=' ', marker='X',
            markersize=10,
            label='velocity, measured, outliers')
    ax.plot(t, v_mov, color=colors[1],
            label='velocity, gaussian weighted moving average, {} samples'.format(
                velocity_window_size))
    ax.plot(t, v_kal, color=colors[3],
            label='velocity, kalman filter')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('velocity [m/s]')
    ax.axhline(0, color='black', linewidth=1, zorder=1)
    ax.legend()

    ax = axes[1]
    ax.plot(t, a, color=colors[5],
            label='deceleration, measured')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('deceleration [m/s^2]')
    ax.axhline(0, color='black', linewidth=1, zorder=1)
    ax.legend()


def plot_vel_fft(r, velocity_window_size):
    charcoal_color = sns.xkcd_palette(['charcoal'])[0]

    t = r['time']
    v = r['speed']
    a = r['accelerometer x']
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
    vf2 = scipy.signal.savgol_filter(v, window_length=111, polyorder=5, deriv=0)

    u = np.reshape(a, (-1, 1, 1))
    z = np.reshape(v, (-1, 1, 1))
    oi = [x for x in util.outlier_index(v, 2*v.std(), 128) if x > 128]

    vf3 = kalman.kalman_velocity(dt, v, u, z, q=5, missed_measurement=oi)

    colors = sns.color_palette('husl', 6)
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(t, v, color=colors[0], alpha=0.5,
            label='velocity, measured')
    ax.plot(t[oi], v[oi], color=colors[0], linestyle=' ', marker='X',
            markersize=10,
            label='velocity, measured, outliers')
    ax.plot(t, vf, color=colors[1],
            label='velocity, gaussian weighted moving average, {} samples'.format(
                velocity_window_size))
    ax.plot(t, vf2, color=colors[3],
            label='velocity, savgol filter (111, 5, 0)')
    ax.plot(t, vf3, color=colors[4],
            label='velocity, kalman filter')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('velocity [m/s]')
    ax.axhline(0, color=charcoal_color, linewidth=1, zorder=1)
    ax.legend()

    freq, xf = ff.fft(v, dt) # uses hamming window
    freq, xf1 = ff.fft(vf, dt) # uses hamming window
    freq, xf2 = ff.fft(vf2, dt) # uses hamming window
    freq, xf3 = ff.fft(vf3, dt) # uses hamming window

    ax = axes[1]
    ax.plot(freq, xf, color=colors[0], alpha=0.5)
    ax.plot(freq, xf1, color=colors[1])
    ax.plot(freq, xf2, color=colors[3])
    ax.plot(freq, xf3, color=colors[4])
    ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('amplitude')
    return fig, axes


def get_trajectory(r, yaw_rate_window_size,
                  plot=False, trial_id=None):
    charcoal_color = sns.xkcd_palette(['charcoal'])[0]

    t = r['time']
    v = r['speed']
    a = r['accelerometer x']
    dt = np.diff(t).mean()
    u = np.reshape(a, (-1, 1, 1))
    z = np.reshape(v, (-1, 1, 1))
    oi = [x for x in util.outlier_index(v, 2*v.std(), 128) if x > 128]

    vf = kalman.kalman_velocity(dt, v, u, z, q=5, missed_measurement=oi)

    yf = ff.moving_average(r['gyroscope z'],
                           yaw_rate_window_size,
                           yaw_rate_window_size/2)

    imu = MadgwickAHRS(sampleperiod=1/125)
    madgwick_roll = []
    madgwick_pitch = []
    madgwick_yaw = []
    for i in range(len(r)):
        imu.update_imu(
                [r[i]['gyroscope x'],
                 r[i]['gyroscope y'],
                 r[i]['gyroscope z']],
                [r[i]['accelerometer x'],
                 r[i]['accelerometer y'],
                 r[i]['accelerometer z']])
        q1, q2, q3 = imu.quaternion.to_euler123() # TODO: Verify this
        madgwick_roll.append(q1)
        madgwick_pitch.append(q2)
        madgwick_yaw.append(q3)

    if plot:
        colors = sns.color_palette('husl', 8)
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes = axes.ravel()

        for ax, c, sig, sigf, ws, l in zip(axes,
                               [colors[1], colors[7]],
                               [r['speed'], r['gyroscope z']],
                               [vf, yf],
                               [0, yaw_rate_window_size],
                               [['velocity', '[m/s]'],
                                ['yaw rate', '[rad/s]']]):
            if ws > 0:
                label = '{}, gaussian weighted moving average, {} samples'.format(
                        l[0], ws)
            else:
                label = 'velocity, kalman filter estimate'
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

    madgwick_yaw = np.array(madgwick_yaw)
    madgwick_yaw -= madgwick_yaw.mean()
    def func2(y, t):
        xp, yp = y
        i = np.argmax(r['time'] >= t)
        v = vf[i]
        yaw = madgwick_yaw[i]
        dydt = [v*np.cos(yaw), v*np.sin(yaw)]
        return dydt
    soln2 = scipy.integrate.odeint(func2, [0, 0], r['time'])

    if plot:
        fig2, axes2 = plt.subplots(2, 1)
        x = soln[:, 0]
        y = soln[:, 1]
        yaw = soln[:, 2]
        x2 = soln2[:, 0]
        y2 = soln2[:, 1]

        ax = axes2[0]
        ax.plot(t, yaw, color=colors[7], label='yaw angle')
        ax.plot(t, madgwick_yaw, color=colors[6], alpha=0.8,
                label='yaw angle, Madgwick IMU filter')
        ax.axhline(0, color=charcoal_color, linewidth=1, zorder=1)
        ax.legend()

        ax = axes2[1]
        ax.plot(x, y, color=colors[7], label='trajectory')
        ax.plot(x2, y2, color=colors[6], label='trajectory (madgwick yaw)')
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
