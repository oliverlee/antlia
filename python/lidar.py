# -*- coding: utf-8 -*-
import glob
import os
import pickle
import numpy as np
import numpy.testing as npt
import scipy.signal

import matplotlib.pyplot as plt
import seaborn as sns

from record import load_file
from util import reduce_runs
from filter import moving_average


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

__record_path = os.path.join(os.path.dirname(__file__),
                             r'../data/lidar/')

def __get_record_files(extension):
    record_path = os.path.join(os.path.dirname(__file__),
                               r'../data/lidar/')
    return sorted(glob.glob('{}*{}'.format(record_path, extension)))


LIDAR_RECORD_FILES =  __get_record_files('.bin')
BICYCLE_RECORD_FILES = __get_record_files('.csv')


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
                                    r'config.p')
    with open(calibration_path, 'rb') as f:
        calibration = pickle.load(f)

    records = []
    for filename in BICYCLE_RECORD_FILES:
        r = load_file(filename, calibration['convbike'])
        records.append(r)
    return records


class Record(object):
    kinds = ('lidar', 'bicycle')
    VELOCITY_FILTER_WINDOW_SIZE = 55

    def __init__(self, lidar_record, bicycle_record):
        self.lidar = lidar_record
        self.bicycle = bicycle_record
        self.synced = None
        self.trials = None

        # filter bicycle speed
        self.bicycle.speed = moving_average(
                self.bicycle.speed,
                self.VELOCITY_FILTER_WINDOW_SIZE,
                self.VELOCITY_FILTER_WINDOW_SIZE/3)

    @staticmethod
    def _nearest_millisecond(x):
        return np.round(x, 3)

    def sync(self):
        if self.synced is None:
            dt = np.diff(self.bicycle.time)
            period = self._nearest_millisecond(dt.mean())
            a = SampledTimeSignal.from_record(self, 'lidar', 'sync', period)
            b = SampledTimeSignal.from_record(self, 'bicycle', 'sync', period)

            sync_offset = a.sync_offset(b)
            time_offset = self.bicycle.time[0] - self.lidar.time[0]
            self.bicycle.time += sync_offset - time_offset

            self.synced = sync_offset
        return self.synced

    def calculate_trials(self):
        #if self.trials is not None: TODO
        #    return

        if self.trials is None:
            rising_edges = np.where(np.diff(self.bicycle.sync) > 0)[0]
            trials = zip(rising_edges, rising_edges[1:])
            t = self.bicycle.time

            # filter out trials that are too short
            MINIMUM_TRIAL_DURATION = 30 # seconds
            trials = [(a, b) for a, b in trials
                             if (t[b] - t[a]) > MINIMUM_TRIAL_DURATION]

            self.trials = list(trials)
        return self.trials


    def plot_timeseries(self, trial=None, timerange=None, ax=None, **kwargs):
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
            _, ax = plt.subplots(4, 1, sharex=True, **kwargs)

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

        t = self.bicycle['time']
        idx = slice(0, -1)
        xlim = None
        if timerange is not None:
            idx = (t >= timerange[0]) & (t < timerange[1])
            xlim = timerange
        elif trial is not None:
            assert(self.trials is not None)
            i0, i1 = self.trials[trial]
            idx = slice(i0, i1)
            xlim = t[i0], t[i1]
        t = t[idx]

        next(colors_iter)
        y = self.bicycle['steer angle'][idx]
        ax[2].plot(t, y, color=next(colors_iter), label='resampled steer angle')
        ax[2].set_xlabel('time [s]')
        ax[2].set_ylabel('steer angle [rad]')
        ax[2].legend()

        next(colors_iter)
        y = self.bicycle['speed'][idx]
        ax[3].plot(t, y, color=next(colors_iter), label='resampled speed')
        ax[3].set_xlabel('time [s]')
        ax[3].set_ylabel('speed [m/s]')
        ax[3].legend()

        if xlim is not None:
            ax[0].set_xlim(xlim)
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
            PREPEND_TIME = 1*60/self.period # 1 minute(s)

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
