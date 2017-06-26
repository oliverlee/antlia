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
    colors = sns.color_palette('Paired', 12)
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
    ax.plot(t, mod_steer, color=colors[1], alpha=0.4,
            label='steer, mean subtracted {:0.2f}, scale factor {:0.2f}'.format(
                steer.mean(), scale))
    ax.plot(t, filt(steer), color=colors[1],
            label=('steer, '
                'highpass butter fc {:0.2f} Hz order {}, '
                'lowpass butter fc {:0.2f} Hz order {}').format(
                       lowcut, order, highcut, order))
    error = filt_steer - mod_steer
    ax.plot(t, error, color=colors[3], alpha=0.2,
            label='error between measured and filtered steer angle')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('steer angle [rad]')

    event_groups = get_steer_event_indices(filt_steer)
    first_turn = True
    for event_range in reversed(event_groups):
        for r0, r1 in event_range:
            sum_error = sum(error[r0:r1])
            sum_filt = sum(filt_steer[r0:r1])

            # if error ratio is too high, discard steering event
            if sum_error/sum_filt < 0.7:
                if first_turn:
                    alpha = 0.4
                    first_turn = False
                    amplitude = np.abs(filt_steer[r0:r1]).max()
                    period = 2*(t[r1] - t[r0])
                    ax.plot(t[r0:r1],
                            (np.sign(filt_steer[int(r0 + r1)//2])*
                                amplitude*np.sin(2*np.pi/period*
                                    (t[r0:r1] - t[r0]))),
                            color=colors[9],
                            label=(
                                'sinusoid fit, '
                                'amplitude {:0.2f}, period {:0.2f}'.format(
                                    amplitude, period)))
                else:
                    alpha = 0.2
                ax.axvspan(t[r0], t[r1], color=colors[5], alpha=alpha)
            else:
                ax.axvspan(t[r0], t[r1], color=colors[7], alpha=0.1)
    ax.legend()
    return fig, ax


def get_steer_event_indices(filt_steer):
    # identify steering event
    sigma = filt_steer.std()
    steer_event_indices = np.argwhere(np.abs(filt_steer) > sigma)
    event_range = util.get_contiguous_numbers(steer_event_indices)
    zero_crossings = np.insert(
            np.array([0, len(filt_steer)]),
            1,
            np.squeeze(np.argwhere(np.diff(np.sign(filt_steer)))))

    # expand ranges to nearest zero crossing
    merged_range = []
    while event_range:
        r0, r1 = event_range[0]
        z0, z1 = zero_crossings[:2]
        assert r0 < r1, 'invalid range'
        assert z0 < z1, 'zero crossings out of order'

        if z0 <= r0 and z1 >= r1:
            if merged_range and (merged_range[-1][-1][1] == z0):
                merged_range[-1].append((z0, z1))
            else:
                merged_range.append([(z0, z1)])
            event_range.pop(0)
            zero_crossings = zero_crossings[1:]
        elif z0 >= r0 and z0 <= r1:
            assert False, 'should not have zero crossing within range'
        elif z1 >= r0 and z1 <= r1:
            assert False, 'should not have zero crossing within range'
        elif z1 < r0:
            zero_crossings = zero_crossings[1:]
        elif z0 > r1:
            zero_crossings = zero_crossings[2:]
        else:
            assert False, 'unhandled case'

    return merged_range
