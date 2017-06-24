#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.lines
import matplotlib.pyplot as plt
import seaborn as sns

import filter as ff
import util


def plot_fft(rec, k_largest=None, max_freq=None):
    util.check_valid_record(rec)
    colors = sns.color_palette('Paired', 6)
    base_color = colors[1]
    k_color = colors[5]

    dt = np.diff(rec['time']).mean()
    # uses hamming window
    freq, xf = ff.fft(rec['steer angle'], dt)

    if max_freq is None:
        max_index = len(rec)
    else:
        max_index = next(x for x in range(len(freq)) if freq[x] >= max_freq)

    if k_largest is None:
        k_largest_freq = None
    else:
        k_indices = sorted(np.argpartition(xf, -k_largest)[-k_largest:])
        msg =  '{}th largest element at freq {} Hz'.format(
                k_largest, freq[k_indices[-1]])
        assert k_indices[-1] <= max_index, msg
        k_largest_freq = freq[k_indices]

    indices = slice(0, max_index)
    fig, ax = plt.subplots()
    markerline, stemline, baseline = ax.stem(freq[indices],
                                             xf[indices],
                                             markerfmt=' ')
    plt.setp(markerline, 'color', base_color)
    plt.setp(stemline, 'color', base_color)

    if k_largest is not None:
        markerline, stemline, baseline = ax.stem(freq[k_indices],
                                                xf[k_indices],
                                                markerfmt=' ')
        plt.setp(markerline, 'color', k_color)
        plt.setp(stemline, 'color', k_color)
        proxy = matplotlib.lines.Line2D([], [], color=k_color)
        ax.legend([proxy],
                  ['{} largest frequency components'.format(k_largest)])

    ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('amplitude')
    return fig, ax, k_largest_freq


def plot_bandpass(rec, lowcut, highcut, order=6):
    colors = sns.color_palette('Paired', 6)
    fs = np.round(1/np.diff(rec['time']).mean())

    from scipy.signal import butter, filtfilt
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    bh, ah = butter(order, low, btype='highpass')
    bl, al = butter(order, high, btype='lowpass')
    filt = lambda x: filtfilt(bh, ah, filtfilt(bl, al, x))

    fig, ax = plt.subplots()
    t = rec['time']
    steer = rec['steer angle']
    filt_steer = filt(steer)
    mod_steer = steer - steer.mean()
    scale = np.abs(filt_steer).mean() / np.abs(mod_steer).mean()
    mod_steer *= scale
    ax.plot(t, mod_steer, alpha=0.3, color=colors[1],
            label='steer, mean subtracted {:0.2f}, scale factor {:0.2f}'.format(
                steer.mean(), scale))
    ax.plot(t, filt(steer), color=colors[1],
            label=('steer, '
                'highpass butter fc {:0.2f} Hz order {}, '
                'lowpass butter fc {:0.2f} Hz order {}').format(
                       lowcut, order, highcut, order))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('steer angle [rad]')
    ax.legend()
    return fig, ax
