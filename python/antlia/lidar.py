# -*- coding: utf-8 -*-
import glob
import os
import pickle
import numpy as np
import numpy.testing as npt
import scipy.signal

import matplotlib.pyplot as plt
import seaborn as sns

from antlia.record import load_file
from antlia.util import reduce_runs
from antlia.filter import moving_average, fft


LIDAR_NUM_ANGLES = 1521
LIDAR_FOV_DEG = 190
LIDAR_SAMPLE_RATE = 20
LIDAR_ANGLES = np.linspace( # in radians
    (90 - LIDAR_FOV_DEG/2)*np.pi/180,
    (90 + LIDAR_FOV_DEG/2)*np.pi/180,
    LIDAR_NUM_ANGLES
)

"""LIDAR datatype format is:
    (
        timestamp (long),
        flag (bool saved as int),
        accelerometer[3] (double),
        gps[3] (double),
        distance[LIDAR_NUM_ANGLES] (long),
    )

    'int' and 'long' are the same size on the raspberry pi (32 bits).
"""
LIDAR_RECORD_DTYPE = np.dtype(','.join(
    ['i4'] + ['i4'] + 3*['f8'] + 3*['f8'] + LIDAR_NUM_ANGLES*['i4']
    )
)
LIDAR_CONVERTED_DTYPE = np.dtype([
    ('time', 'f8'),
    ('sync', 'f8'),
    ('accelerometer x', 'f8'), # body x-axis may not be aligned with inertial!
    ('accelerometer y', 'f8'), # body y-axis may not be aligned with inertial!
    ('accelerometer z', 'f8'), # body z-axis may not be aligned with inertial!
    ('gps', 'f8', (3,)),
    ('distance', 'f8', (LIDAR_NUM_ANGLES,)),
])


def __get_record_files(extension):
    record_path = os.path.join(os.path.dirname(__file__),
                               r'../../data/lidar/')
    return sorted(glob.glob('{}*{}'.format(record_path, extension)))


LIDAR_RECORD_FILES =  __get_record_files('.bin')
BICYCLE_RECORD_FILES = __get_record_files('.csv')
assert LIDAR_RECORD_FILES, "No LIDAR records found!"
assert BICYCLE_RECORD_FILES, "No bicycle records found!"


def _get_lidar_records(convert_dtype=True):
    """Returns a list of LIDAR records.

    Parameters:
    convert_dtype: bool, Datatype of LIDAR records.
                   True    - dtype = LIDAR_CONVERTED_DTYPE
                   False   - dtype = LIDAR_RECORD_DTYPE
    Returns:
    records: list of array_like
    """
    float_dtype = np.dtype(','.join(len(LIDAR_RECORD_DTYPE) * ['f8']))

    records = []
    for filename in LIDAR_RECORD_FILES:
        with open(filename, 'rb') as f:
            x = np.fromfile(f, LIDAR_RECORD_DTYPE)
            if convert_dtype:
                # copy and convert data to new dtype
                y = x.astype(float_dtype)
                # create view which shares the same underlying memory
                x = y.view(LIDAR_CONVERTED_DTYPE).view(np.recarray)
                # convert timestamp to time and start at zero
                x.time /= 1000
                x.time -= x.time[0]
                # flip sync value to be active high
                x.sync = np.invert(x.sync.astype(bool)).astype(x.sync.dtype)
                # TODO: convert accelerometer
                # TODO: convert gps
                # convert distance from millimeters to meters
                x.distance /= 1000
            records.append(x)
    return records


def _get_bicycle_records():
    # load calibration data
    calibration_path = os.path.join(os.path.dirname(__file__),
                                    '..', r'config.p')
    with open(calibration_path, 'rb') as f:
        calibration = pickle.load(f)

    records = []
    for filename in BICYCLE_RECORD_FILES:
        r = load_file(filename, calibration['convbike'])
        records.append(r)
    return records


class Record(object):
    kinds = ('lidar', 'bicycle')

    def __init__(self, lidar_record, bicycle_record):
        self.lidar = lidar_record
        self.bicycle = bicycle_record
        self.synced = None
        self._trial = None
        self._trial_range_index = None

        dt = np.diff(self.bicycle.time)
        self.bicycle_period = self._nearest_millisecond(dt.mean())

    @staticmethod
    def _nearest_millisecond(x):
        return np.round(x, 3)

    def sync(self):
        if self.synced is None:
            period = self.bicycle_period
            a = SampledTimeSignal.from_record(self, 'lidar', 'sync', period)
            b = SampledTimeSignal.from_record(self, 'bicycle', 'sync', period)

            sync_offset = a.sync_offset(b)
            time_offset = self.bicycle.time[0] - self.lidar.time[0]
            self.bicycle.time += sync_offset - time_offset

            self.synced = sync_offset
        return self.synced

    def calculate_trials(self):
        if self._trial is None:
            rising_edges = np.where(np.diff(self.bicycle.sync) > 0)[0]
            trials = zip(rising_edges, rising_edges[1:])
            t = self.bicycle.time

            # filter out trials that are too short
            MINIMUM_TRIAL_DURATION = 30 # seconds
            self._trial = [self.bicycle[a:b]
                           for a, b in trials
                           if (t[b] - t[a]) > MINIMUM_TRIAL_DURATION]

            trial_range_index = []
            for i in range(len(self._trial)):
                trial_range_index.append(self._calculate_trial_ranges(i))
            self._trial_range_index = trial_range_index

        return self._trial

    def trials(self, trial_range=4):
        valid = lambda i: self._trial_range_index[i][4] is not None
        return (self.trial(i, trial_range)
                if valid(i) else None
                for i in range(len(self._trial)))


    def trial(self, trial_number, trial_range=5):
        if self._trial is None:
            self.calculate_trials()

        if trial_range < 5:
            extrema = self._trial_range_index[trial_number][trial_range]
            index = slice(*extrema[np.array([0, -1])])
            return self._trial[trial_number][index]
        elif trial_range == 5:
            try:
                return self._active_trial_range(trial_number)
            except TypeError:
                # missing/problematic velocity signal for rider 0 trial {0, 1}
                # use manually calculated values

                # assume this error only occurs for rider 0
                trial = self._trial[trial_number]
                if trial_number == 0:
                    index = (trial.time > 80) & (trial.time < 95)
                elif trial_number == 1:
                    index = (trial.time > 200) & (trial.time < 280)
                return trial[index]

        raise ValueError(
            'Invalid value for \'trial_range\', {}'.format(trial_range))

    def _active_trial_range(self, trial_number):
        """Extract the range of a trial where the participant is active. This
        removes the parts of the start and end of a trial and may not capture
        initial acceleration nor final deceleration.
        """
        MIN_SPEED = 2.8 # m/s

        assert self._trial is not None
        trial = self.trial(trial_number, 4)

        # filter speed
        v = self._cheby1_lowpass_filter(trial.speed, 0.5)

        edges = np.diff((v > MIN_SPEED).astype(int))
        edge_index = np.where(edges)[0]
        edge_type = edges[edge_index] # rising (1) or falling (-1)

        n = len(edge_type)
        assert n > 0 # verify the signal exceeds MIN_SPEED

        # go through observed cases
        if n == 1 and edge_type[0] == -1:
            # single falling edge in the second half of trial
            # set rising edge to the first sample
            assert edge_index[0] > n/2
            index = np.insert(edge_index, 0, 0)
        elif n == 2 and edge_type[0] == 1 and edge_type[1] == -1:
            # normal case with single rising edge and single falling edge
            index = edge_index
        elif n == 3 and edge_type[0] == -1 and edge_index[0] < 100:
            # extra falling edge at the start of the trial
            assert edge_type[1] == 1
            assert edge_type[2] == -1
            index = edge_index[1:]
        else:
            # some other non-handled case occurs
            raise NotImplementedError
        return trial[slice(index[0], index[1])]

    def _cheby1_lowpass_filter(self, x, fc):
        period = self.bicycle_period
        fs = 1/period # bicycle sample rate
        order = 5
        apass = 0.001 # dB

        wn = fc / (0.5*fs)
        b, a = scipy.signal.cheby1(order, apass, wn)
        return scipy.signal.filtfilt(b, a, x)

    def _calculate_trial_ranges(self, trial_number):
        """In a trial, we normally observe 3 phases. We describe each phase as:
        1. Subject moves from the LIDAR/obstacle to path start
        2. Subject moves from path start to path end
        3. Subject moves from path end to the LIDAR/obstacle

        Each phase corresponds to a bump in the speed signal. Phase 2 is
        extracted by the following method:
        1. Apply a low-pass filter to the speed signal.
        2. Calculate local extrema of speed signal for the entire trial.
           This is denoted as RANGE0.
        3. Pick local maxima from RANGE0 that exceed the threshold RANGE1_LIMIT.
           The time spanned by these local maxima denote RANGE1.
        4. Pick local minima within RANGE1 and less than
           RANGE2_PERCENT*min(maxima_RANGE1) and RANGE2_LIMIT. The time
           spanned by these local minima denote RANGE2.
        5. Pick local maxima within RANGE2. The time spanned by these local
           maxima denote RANGE3.
        6. Find the local minima of RANGE2 closest in time before and after
           RANGE3.
        """
        RANGE1_LIMIT = 1 # [kph]
        RANGE2_LIMIT = 4 # [kph]
        RANGE2_PERCENT = 0.9 # [percent]

        # filtered speed, range0 extrema
        v = self._cheby1_lowpass_filter(self._trial[trial_number].speed, 0.08)
        r0_minima = scipy.signal.argrelextrema(v, np.less)[0]
        r0_maxima = scipy.signal.argrelextrema(v, np.greater)[0]
        r0 = np.concatenate((r0_minima, r0_maxima))

        r1_maxima = np.array([i for i in r0_maxima if v[i] > RANGE1_LIMIT])
        if len(r1_maxima) < 1:
            print('Unable to determine range 1')
            return (r0, None, None, None, None)

        a = min(RANGE2_PERCENT * v[r1_maxima].min(), RANGE2_LIMIT)
        r2_minima = np.array([i for i in r0_minima
                              if ((i > r1_maxima[0]) and
                                  (i < r1_maxima[-1]) and
                                  (v[i] < a))])
        if len(r2_minima) < 1:
            print('Unable to determine range 2')
            return (r0,
                    r1_maxima,
                    None,
                    None,
                    None)

        ia = r1_maxima > r2_minima[0]
        ib = r1_maxima < r2_minima[-1]
        r3_maxima = r1_maxima[ia & ib]

        b = None
        c = None
        for i in r2_minima:
            if i < r3_maxima[0]:
                b = i
            if i > r3_maxima[-1]:
                c = i
                break
        # Special case to fix range 4 for rider 3, trial 0.
        for i in r2_minima:
            if i > b and i < c:
                b = i
                break;
        r4 = np.array([b, c])

        return (r0,
                r1_maxima,
                r2_minima,
                r3_maxima,
                r4)

    def _chebwin_fft(self, x, attenuation=300, max_freq=None):
        """Calculate the FFT of x with a chebwin window.

        Parameters:
        x: array_like, signal
        attenuation: float, attenuation of Dolph-Chebyshev window in dB
        max_freq: float, upper limit for returned frequency vector

        Returns:
        freq: array_like, frequencies
        xf: array_like, frequency component of signal 'x'
        """
        window = lambda x: scipy.signal.chebwin(x, at=attenuation, sym=False)
        freq, xf = fft(x, self.bicycle_period, window)
        if max_freq is not None:
            index = freq < max_freq
        else:
            index = slice(0, None)
        return freq[index], xf[index]

    def steer_angle_cutoff(self, trial, intermediate_values=False):
        freq, xf = self._chebwin_fft(trial['steer angle'], max_freq=2)

        # find local minima in FFT
        minima = scipy.signal.argrelextrema(xf, np.less)[0]
        m0 = minima[0]
        m1 = minima[1]
        m2 = minima[2]

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

    def _butter_bandpass_filter(self, x, fc):
        fs = 1/self.bicycle_period
        order = 3
        wn = np.array([0.1, fc]) / (0.5*fs)
        b, a = scipy.signal.butter(order, wn, btype='bandpass')
        return scipy.signal.filtfilt(b, a, x)

    def filtered_steer_angle(self, trial, fc=None):
        if fc is None:
            fc = self.steer_angle_cutoff(trial)

        return self._butter_bandpass_filter(trial['steer angle'], fc)

    def plot_steer_angle_filter_calculation(self, trial_number,
                                            ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(2, 1, **kwargs)

        colors = sns.color_palette('Paired', 10)

        trial = self.trial(trial_number)
        cutoff, (freq, xf), minima = self.steer_angle_cutoff(trial, True)
        m0 = minima[0]
        m1 = minima[1]
        m2 = minima[2]

        t = trial.time
        x = trial['steer angle']
        z_06 = self._butter_bandpass_filter(x, 0.6)
        z_m1 = self._butter_bandpass_filter(x, freq[m1])
        z_m2 = self._butter_bandpass_filter(x, freq[m2])
        z_c = self._butter_bandpass_filter(x, cutoff)

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
        ax[1].plot(*self._chebwin_fft(z_06, max_freq=max_freq),
                   label='steer angle (fc = 0.6 Hz)',
                   color=colors[1], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_m1, max_freq=max_freq),
                   label='steer angle (fc = {:0.02f} Hz), m1'.format(freq[m1]),
                   color=colors[5], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_m2, max_freq=max_freq),
                   label='steer angle (fc = {:0.02f} Hz), m2'.format(freq[m2]),
                   color=colors[7], alpha=0.6)
        ax[1].plot(*self._chebwin_fft(z_c, max_freq=max_freq), '--',
                   label='steer angle (fc = {:0.02f} Hz), c'.format(cutoff),
                   color=colors[9])

        # plot this FFT last and use saved ylim as this dominates due to
        # the DC  component
        ylim = ax[1].get_ylim()
        ax[1].plot(*self._chebwin_fft(x, max_freq=max_freq),
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
        return ax

    def plot_timing(self, ax=None, **kwargs):
        def plot_two(ax, data, color, label):
            call = lambda f: tuple(map(f, self.kinds))

            if callable(data):
                data = call(data)
            if callable(color):
                color = call(color)
            if callable(label):
                label = call(label)

            args = zip(data, color, label)
            return [ax.plot(*d, color=c, label=l) for d, c, l in args]

        if ax is None:
            _, ax = plt.subplots(2, 1, sharex=True, **kwargs)

        colors = sns.color_palette('Paired', 10)
        colors_iter = iter(colors)

        def reduced_signal(t, x):
            xr, index = reduce_runs(x)
            return t[index], xr

        def get_sample_time(key):
            time = getattr(self, key)['time']
            sample_time = self._nearest_millisecond(np.diff(time))
            return reduced_signal(time[1:], sample_time)
        plot_two(ax[0],
                 get_sample_time,
                 colors_iter,
                 lambda k: '{} sample time'.format(k))
        ax[0].set_xlabel('time [s]')
        ax[0].set_ylabel('last sample time [s]')
        ax[0].legend()

        def sync_func(key):
            time = getattr(self, key)['time']
            sync = getattr(self, key)['sync']
            return reduced_signal(time, sync)
        plot_two(ax[1],
                 sync_func,
                 colors_iter,
                 lambda k: '{} sync button'.format(k))
        ax[1].set_xlabel('time [s]')
        ax[1].set_ylabel('button status')
        ax[1].legend()
        return ax

    def plot_trial_range_calculation(self, trial_number, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(**kwargs)

        colors = sns.color_palette('Paired', 10)

        ran = self._trial_range_index[trial_number]
        trial = self._trial[trial_number]
        v = self._cheby1_lowpass_filter(trial.speed)

        ax.plot(trial.time, trial.speed, label='speed',
                alpha=0.5, color=colors[0], zorder=0)
        ax.plot(trial.time, v, label='speed, cheby1 low pass',
                linewidth=3, color=colors[2], zorder=1)
        try:
            # use trial() method to get data sliced to valid range
            valid_trial = self.trial(trial_number)
            ax.plot(valid_trial.time, valid_trial.speed,
                    label='speed (valid range)',
                    color=colors[1], zorder=0)
        except TypeError:
            pass

        mt = 'X' # markertype
        ms = 10 # markersize
        ax.plot(trial.time[ran[0]], v[ran[0]], mt, label='range 0 extrema',
                markersize=ms, color=colors[3])
        if ran[1] is not None:
            ax.plot(trial.time[ran[1]], v[ran[1]], mt, label='range 1 maxima',
                    markersize=ms, color=colors[7])
        if ran[2] is not None:
            ax.plot(trial.time[ran[2]], v[ran[2]], mt, label='range 2 minima',
                    markersize=ms, color=colors[5])
        if ran[4] is not None:
            ax.plot(trial.time[ran[4]], v[ran[4]], mt, label='range 4 minima',
                    markersize=ms, color=colors[9])
            ax.axvspan(trial.time[ran[4]][0], trial.time[ran[4]][-1],
                       label='range 4',
                       alpha=0.5, color=colors[8])

        ax.set_xlabel('time [s]')
        ax.set_ylabel('speed [m/s]')
        ax.legend()
        return ax


def load_records(sync=False, calculate_trials=True):
    records = [Record(l, b) for l, b in zip(_get_lidar_records(),
                                            _get_bicycle_records())]
    for r in records:
        if sync:
            r.sync()
        if calculate_trials:
            r.calculate_trials()
    return records


class TimeSignal(object):
    def __init__(self, time, signal):
        assert(isinstance(time, np.ndarray))
        assert(isinstance(signal, np.ndarray))
        assert(time.shape == signal.shape)
        self.time = time
        self.signal = signal
        self.time.flags.writeable = False
        self.signal.flags.writeable = False
        self.__mutable = False

    def shift_time(self, shift):
        if self.__mutable:
            self.time.flags.writeable = True
            self.time += shift
            self.time.flags.writeable = False
            return True
        return False


class SampledTimeSignal(TimeSignal):
    def __init__(self, time, signal, period=None):
        super(SampledTimeSignal, self).__init__(time, signal)
        self.period = np.diff(time).mean()
        if period is not None:
            npt.assert_almost_equal(self.period, period)

        # allow methods to modify time and signal data
        self.__mutable = True

    def shift_time_index(self, shift):
        return self.shift_time(shift * self.period)

    def __getitem__(self, key):
        period = self.period
        if key.step is not None:
            period *= key.step

        signal_slice = SampledTimeSignal(self.time[key], self.signal[key], period)
        signal_slice.__mutable = False
        return signal_slice

    @classmethod
    def from_record(cls, record, kind, signal, period):
        """Create a SampledTimeSignal from a specific signal in a Record. This
        will resample the signal with the given period by interpolation.

        Parameters
        record: Record, source data
        kind: string, 'lidar' or 'bicycle'
        signal: string, name of signal
        period: float, resampled signal period
        """
        record_kind = getattr(record, kind)
        resampled_time = np.arange(0, record_kind['time'].max(), period)
        resampled_signal = np.interp(resampled_time,
                                     record_kind['time'],
                                     record_kind[signal])

        return cls(resampled_time, resampled_signal, period)

    def sync_offset(self, other):
        """Calculate the time offset of 'other' from 'self' by maximizing
        signal cross-correlation.

        Parameters:
        other: SampledTimeSignal

        Returns:
        offset: float, value in seconds

        Notes:
        self.period and other.period must be equal.
        """
        assert(isinstance(other, type(self)))
        npt.assert_almost_equal(self.period, other.period)

        time_offset = other.time[0] - self.time[0]

        # If other signal is longer, np.correlate() will swap arguments.
        if (other.signal.shape[0] > self.signal.shape[0]):
            return -other.sync_offset(self)

        c = np.correlate(self.signal, other.signal, mode='valid')

        # If synchronization failure, prepend data to longer signal and try
        # again. This is much faster than using 'full' mode in np.correlate().
        sync_success_limit = 0.9 * self.signal.sum();
        index = c.argmax()
        if c[index] < sync_success_limit:
            PREPEND_TIME = 10/self.period # 10 second(s)

            start = self.time[0] - self.period
            stop = start - PREPEND_TIME
            n = -(stop - start)/self.period
            npt.assert_almost_equal(n, np.round(n))
            n = int(n)

            c = np.correlate(np.concatenate((np.zeros((n,)), self.signal)),
                             other.signal,
                             mode='valid')

            time_offset += self.time[0] - stop

        index = c.argmax()
        if c[index] < sync_success_limit:
            raise ValueError(
                    'Unable to synchronize signals {}, {}',
                    self, other)

        return self.period*index - time_offset
