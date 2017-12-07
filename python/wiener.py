#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

from antlia import filter as ff
from antlia import plot_braking as braking
from antlia import plot_steering as steering
from antlia import path
from antlia import record
from antlia import util


def load_calibration_records(speed=10):
    config_path = os.path.join(
            os.path.dirname(__file__),
            'config.p')
    with open(config_path, 'rb') as f:
        cd = pickle.load(f)

    path = os.path.join(
            os.path.dirname(__file__),
            r'../data/etrike/calibration/convbike/speed/{}kph.csv'.format(speed))
    return record.load_file(path, cd['convbike'])


def plot_velocity(r):
    charcoal_color = sns.xkcd_palette(['charcoal'])[0]

    t = r['time']
    v = r['speed']
    dt = np.diff(t).mean()

    # from visual inspection of plot
    i0 = np.argwhere(t > 20)[0][0]
    i1 = np.argwhere(t > 120)[0][0]
    vx = v[i0:i1].mean()

    sg_window_length = 255
    sg_polyorder = 2
    vf = scipy.signal.savgol_filter(v, sg_window_length, polyorder=sg_polyorder)
    wiener_length = 256
    h = wiener_taps(vf, v, wiener_length)
    vf2 = scipy.signal.filtfilt(h, np.array([1]), v)
    vf3 = scipy.signal.lfilter(h, np.array([1]), v)

    vf4 = ff.moving_average(v, wiener_length - 1, wiener_length/2)

    colors = sns.color_palette('Paired', 10)
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(t[i0:i1], vx*np.ones(i1 - i0),
            color=colors[1], alpha=1, zorder=1,
            label='velocity, measurement mean')
    ax.plot(t, v,
            color=colors[3], alpha=1, zorder=0,
            label='velocity, measured')
    ax.plot(t, vf,
            color=colors[5], alpha=1, zorder=0,
            label='velocity, savitzky-golay filter, length={}, order={}'.format(
                sg_window_length, sg_polyorder))
    ax.plot(t, vf2,
            color=colors[7], alpha=1, zorder=2,
            label='velocity, wiener filter, length={}, filtfilt'.format(wiener_length))
    ax.plot(t, vf3,
            color=colors[6], alpha=1, zorder=2,
            label='velocity, wiener filter, length={}, lfilter'.format(wiener_length))
    ax.plot(t, vf3,
            color=colors[8], alpha=1, zorder=2,
            label='velocity, moving average, length={}'.format(wiener_length - 1))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('velocity [m/s]')
    ax.axhline(0, color=charcoal_color, linewidth=1, zorder=1)
    ax.legend()

    freq, xf0 = ff.fft(v, dt) # uses hamming window
    freq, xf1 = ff.fft(vf, dt) # uses hamming window
    freq, xf2 = ff.fft(vf2, dt) # uses hamming window

    ax = axes[1]
    ax.plot(freq, xf0, color=colors[3], alpha=1, zorder=0)
    ax.plot(freq, xf1, color=colors[5], alpha=1, zorder=0)
    ax.plot(freq, xf2, color=colors[7], alpha=1, zorder=1)
    ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('amplitude')


def wiener_taps(original, measured, filter_length):
    s = original
    x = measured
    M = filter_length

    # estimate cross PSDs using Welch
    f, Pxx = scipy.signal.csd(x, x, nperseg=M)
    f, Psx = scipy.signal.csd(s, x, nperseg=M)
    # compute Wiener filter
    H = Psx/Pxx
    ## shift for causal filter
    H = H*np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2))
    h = np.fft.irfft(H)
    h /= sum(h)
    return h


if __name__ == '__main__':
    r = load_calibration_records()
    plot_velocity(r)

    plt.show()
