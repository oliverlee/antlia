# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

from antlia.filter import fft
from antlia.pattern import ExtremaList, SteerEvent, window


class Trial(object):
    def __init__(self, data, period):
            self.data = data
            self.period = period

    @staticmethod
    def _butter_bandpass_filter(x, fc, fs):
        order = 3
        wn = np.array([0.1, fc]) / (0.5*fs)
        b, a = scipy.signal.butter(order, wn, btype='bandpass')
        return scipy.signal.filtfilt(b, a, x)

    @staticmethod
    def _chebwin_fft(x, dt, attenuation=300, max_freq=None):
        """Calculate the FFT of x with a chebwin window.

        Parameters:
        x: array_like, signal
        dt: float, sample period
        attenuation: float, attenuation of Dolph-Chebyshev window in dB
        max_freq: float, upper limit for returned frequency vector

        Returns:
        freq: array_like, frequencies
        xf: array_like, frequency component of signal 'x'
        """
        window = lambda x: scipy.signal.chebwin(x, at=attenuation, sym=False)
        freq, xf = fft(x, dt, window)
        if max_freq is not None:
            index = freq < max_freq
        else:
            index = slice(0, None)
        return freq[index], xf[index]

    def filtered_steer_angle(self, fc=None):
        if fc is None:
            fc = self.steer_angle_cutoff()

        return self._butter_bandpass_filter(
                self.data['steer angle'], fc, 1/self.period)

    def steer_angle_cutoff(self, intermediate_values=False):
        freq, xf = self._chebwin_fft(self.data['steer angle'],
                                     self.period,
                                     max_freq=2)

        # find local minima in FFT
        minima = scipy.signal.argrelextrema(xf, np.less)[0]
        m0 = minima[0]
        m1 = minima[1]
        try:
            m2 = minima[2]
        except IndexError:
            raise ValueError(
                    'Unable to find m2. Data possibly missing samples.')

        # default cutoff frequency
        cutoff = freq[m1]

        # Handle special cases. The cutoff frequency should maximize frequency
        # components corresponding with maneuvering and minimize frequency
        # components corresponding with pedalling. This appears to be the first
        # bump in the FFT of the steer angle.
        if freq[m0] > 0.5:
            # FFT window main lobe too large and first minimum not detected
            cutoff = 0.6
        elif freq[m1] < 0.5 and xf[m1] > xf[m0]:
            # leakage from nearby frequencies results in an invalid minimum
            for m in minima[2:]:
                if xf[m] < xf[m0]:
                    cutoff = freq[m]
                    break
        elif freq[m1] > 1 and (freq[m1] - freq[m0]) > 1:
            # minimum between m0 and m1 not detected
            cutoff = 0.5*(0.5 + freq[m1])
        elif np.abs(xf[m1] - xf[m0]) < 0.2*xf[m2] and freq[m0] < 0.5:
            # m1 too close to m0, frequencies between the two are attenuated
            # when steer angle is filtered
            cutoff = 0.33*(freq[m1] + freq[m2])

        if intermediate_values:
            return cutoff, (freq, xf), minima
        return cutoff

    def plot_steer_angle_filter_calculation(self, ax=None, **kwargs):
        cutoff, (freq, xf), minima = self.steer_angle_cutoff(True)
        m0 = minima[0]
        m1 = minima[1]
        try:
            m2 = minima[2]
        except IndexError:
            m2 = 18

        t = self.data.time
        dt = self.period
        x = self.data['steer angle']
        z_06 = self._butter_bandpass_filter(x,  0.6, 1/dt)
        z_m1 = self._butter_bandpass_filter(x, freq[m1], 1/dt)
        z_m2 = self._butter_bandpass_filter(x, freq[m2], 1/dt)
        z_c = self._butter_bandpass_filter(x, cutoff, 1/dt)

        if ax is None:
            _, ax = plt.subplots(2, 1, **kwargs)

        colors = sns.color_palette('Paired', 10)

        # plot filtered versions of steer angle signal
        ax[0].plot(t, z_06,
                   label='steer angle (fc = 0.6 Hz)',
                   color=colors[1], alpha=0.6)
        ax[0].plot(t, z_m1,
                   label='steer angle (fc = {:0.02f} Hz), m1'.format(freq[m1]),
                   color=colors[5], alpha=0.6)
        ax[0].plot(t, z_m2,
                   label='steer angle (fc = {:0.02f} Hz), m2'.format(freq[m2]),
                   color=colors[7], alpha=0.6)
        ax[0].plot(t, z_c, '--',
                   label='steer angle (fc = {:0.02f} Hz), c'.format(cutoff),
                   color=colors[9])
        ax[0].plot(t, x,
                   label='steer angle (unfiltered)',
                   color=colors[3])
        ax[0].set_xlabel('time [s]')
        ax[0].set_ylabel('speed [m/s]')
        ax[0].legend()

        # plot FFTs
        max_freq = 5 # Hz
        ax[1].plot(*self._chebwin_fft(z_06, dt, max_freq=max_freq),
                   label='steer angle (fc = 0.6 Hz)',
                   color=colors[1], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_m1, dt, max_freq=max_freq),
                   label='steer angle (fc = {:0.02f} Hz), m1'.format(freq[m1]),
                   color=colors[5], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_m2, dt, max_freq=max_freq),
                   label='steer angle (fc = {:0.02f} Hz), m2'.format(freq[m2]),
                   color=colors[7], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_c, dt, max_freq=max_freq), '--',
                   label='steer angle (fc = {:0.02f} Hz), c'.format(cutoff),
                   color=colors[9])

        # plot this FFT last and use saved ylim as this dominates due to
        # the DC  component
        ylim = ax[1].get_ylim()
        ax[1].plot(*self._chebwin_fft(x, dt, max_freq=max_freq),
                   label='steer angle (unfiltered)',
                   color=colors[3])
        ax[1].set_ylim(ylim)

        # plot minima markers
        ax[1].plot(freq[minima][:10], xf[minima][:10],
                   'X', markersize=10,
                   color=colors[3], alpha=0.6)
        ax[1].plot(freq[m1], xf[m1],
                   'o', markersize=12,
                   color=colors[5], alpha=0.6)
        ax[1].plot(freq[m2], xf[m2],
                   'o', markersize=12,
                   color=colors[7], alpha=0.6)

        ax[1].set_title('steer angle FFT')
        ax[1].set_xlabel('frequency [Hz]')
        ax[1].legend()
        return ax, cutoff, freq[m1], freq[m2]

    def _get_steer_events(self):
        extrema = ExtremaList(self)
        steer_angle = self.filtered_steer_angle()

        events = []
        for w in window(extrema.extrema, len(SteerEvent.pattern)):
            try:
                events.append(SteerEvent(w, self, steer_angle))
            except ValueError:
                pass

        # filter out events with less than 0.1 of max score
        max_score = max(ev.score for ev in events)
        events = [ev for ev in events if ev.score > 0.1*max_score]

        # filter out events with average speed less than 0.9 of max speed
        max_speed = max(ev.speed for ev in events)
        events = [ev for ev in events if ev.speed > 0.9*max_speed]

        return events, extrema, steer_angle

    def plot_steer_event_detection(self, ax=None, **kwargs):
        events, extrema, steer_angle = self._get_steer_events()

        colors = sns.color_palette('Paired', 10)

        if ax is None:
            fig, ax = plt.subplots(**kwargs)

        time = self.data.time
        maxima = extrema.maxima
        minima = extrema.minima
        inflections = extrema.inflections

        ax.plot(time, steer_angle, label='filtered steer angle',
                color=colors[1])
        ax.plot(time[maxima], steer_angle[maxima], 'X', label='maxima',
                color=colors[3])
        ax.plot(time[minima], steer_angle[minima], 'X', label='minima',
                color=colors[5])
        ax.plot(time[inflections], steer_angle[inflections], 'X',
                label='inflection points',
                color=colors[9])
        ax.axhline(0, color='black')
        ax.legend()

        # recalculate max score
        max_score = max(ev.score for ev in events)
        for ev in events:
            t0 = time[ev.start]
            t1 = time[ev.end]
            ax.axvspan(t0, t1, alpha=max(0, ev.score/max_score),
                       color=colors[6])

        ax.set_title(', '.join('({:0.2f}, {:0.2f})'.format(ev.score, ev.speed)
                               for ev in events))
        return ax

    def steer_event_parameters(self):
        events, _, steer_angle = self._get_steer_events()
        ev = max(events)

        def initial_fit_parameters(t, y):
            # best fit sinusoid
            ampl = -np.abs(y).max()
            freq = 1/(2*(t[-1] - t[0]))
            phase = -t[0]
            mean = 0
            return ampl, freq, phase, mean

        def optimize_func(y):
            return lambda x: x[0]*np.sin(2*np.pi*x[1]*(t + x[2])) + x[3] - y

        def estimate_fit_parameters(t, y):
            return scipy.optimize.leastsq(
                    optimize_func(y),
                    initial_fit_parameters(t, y))[0]

        # get indices for first turn in the steer event
        # this corresponds to (inflect, minimum, inflect)
        i0 = ev.extrema[0].index
        i1 = ev.extrema[2].index

        # get corresponding time and steer angle subset
        index = slice(i0 - i0, i1 - i0)
        t = ev.time[index]
        y = ev.steer_angle[index]
        estimate = estimate_fit_parameters(t, y)
        estimate_steer = optimize_func(y)(estimate) + y

        period = 1/estimate[1]
        if period > 6:
            # If period is too large, retry fit with last 75% of data
            n = len(t)//4
            t = t[n:]
            y = y[n:]
            estimate = estimate_fit_parameters(t, y)
            estimate_steer = optimize_func(y)(estimate) + y

        return estimate, (t, y, estimate_steer), ev
