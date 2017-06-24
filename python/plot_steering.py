#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.lines
import matplotlib.pyplot as plt
import seaborn as sns

import filter as ff
import util


def plot_steer_angle_fft(rec, k_largest=None, max_freq=None):
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
