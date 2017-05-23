#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import scipy.fftpack


def moving_average(x, window_size, sigma=None):
    """ Apply a non-causal moving average filter to signal x. Window size must
    be positive and odd. If sigma is None, all samples in the window are
    weighted equally, otherwise a gaussian distribution is used.
    """
    assert window_size > 0 # window size must be positive
    assert window_size % 2 == 1 # window size must be odd
    if sigma is None:
        weights = np.repeat(1.0, window_size) / window_size
    else:
        n = int(np.floor(window_size/2))
        w = np.linspace(-n, n, 2*n + 1)
        weights = np.exp(-w**2 / (2*sigma**2))
        weights /= sum(weights)
    return np.convolve(x, weights, 'same')


def fft(x, sample_period, window_type=None):
    """ Calculate the Fourier transform for signal x. Input is assumed real and
    the first half of the frequencies and amplitudes are returned. window_type
    specifies the windowing function used, hamming is default.
    """
    if window_type is None:
        window_type = scipy.signal.hamming
    n = len(x)
    windowed_x = np.multiply(x, window_type(n))

    # only use first half of fft since real signals are mirrored about nyquist
    # frequency
    xf = 2/n * np.abs(scipy.fftpack.fft(windowed_x)[:n//2])
    freq = np.linspace(0, 1/(2*sample_period), n/2)
    return freq, xf


def rolling_fft(x, sample_period,
                window_start_indices, window_length, window_type=None):
    X = []
    for i in window_start_indices:
        freq, xf = fft(x[i:i + window_length], sample_period, window_type)
        X.append(xf)
    return freq, window_start_indices, X
