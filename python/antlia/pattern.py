from enum import Enum
import collections
import functools
import itertools

import numpy as np
import scipy


@functools.total_ordering
class Extremum(object):
    class Concavity(Enum):
        MINIMUM = -1
        INFLECT =  0
        MAXIMUM =  1

    def __init__(self, index, concavity, value):
        self.index = index
        self.concavity = concavity
        self.value = value

    def __eq__(self, other):
        return self.index == other.index

    def __lt__(self, other):
        return self.index < other.index


def window(iterator, size=2):
    """sliding window iterator
    """

    if not isinstance(iterator, collections.Iterator):
        iterator = iter(iterator)

    result = tuple(itertools.islice(iterator, size))
    if len(result) == size:
        yield result
    for elem in iterator:
        result = result[1:] + (elem,)
        yield result


class ExtremaList(object):
    def __init__(self, trial):
        self.trial = trial

        steer_angle = trial.filtered_steer_angle()
        maxima = scipy.signal.argrelmax(steer_angle)[0]
        minima = scipy.signal.argrelmin(steer_angle)[0]

        steer_rate = np.diff(steer_angle)
        inflections = np.concatenate((
            scipy.signal.argrelmin(steer_rate)[0],
            scipy.signal.argrelmax(steer_rate)[0]))

        extrema = list(zip(maxima,
                           [Extremum.Concavity.MAXIMUM] * len(maxima),
                           steer_angle[maxima]))
        extrema.extend(list(zip(minima,
                                [Extremum.Concavity.MINIMUM] * len(minima),
                                steer_angle[minima])))
        extrema.extend(list(zip(inflections,
                                [Extremum.Concavity.INFLECT] * len(inflections),
                                steer_angle[inflections])))
        extrema.sort() # python tuple sort
        extrema = [Extremum(index, concavity, value)
                   for index, concavity, value in extrema]

        self.steer_angle = steer_angle
        self.maxima = maxima
        self.minima = minima
        self.inflections = inflections # initial inflection point detection
        self.extrema = self._reduce_inflection_points(extrema)

    @staticmethod
    def _reduce_inflection_points(extrema):
        # We often see 3 inflection points between 2 extrema. Reduce 3 to 1.
        removed = set()
        for i, w in enumerate(window(extrema,
                                     len(_inflection_point_pattern))):
            if _inflection_point_pattern.match(w):
                # Keep the point which has the smallest difference in distance
                # to the two endpoints
                difference = []
                for x in w[1:4]:
                    d = (x.index - w[0].index) - (w[-1].index - x.index)
                    difference.append(abs(d))

                l = [0, 1, 2] # corresponds to elements 1, 2, 3 in this window
                l.remove(np.argmin(difference))

                for window_index in l:
                    removed.add(i + 1 + window_index)
        if removed:
            extrema = [e
                       for i, e in enumerate(extrema)
                       if i not in removed]
        return extrema


class Pattern(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __len__(self):
        return len(self.pattern)

    def match(self, window):
        if len(window) != len(self.pattern): raise ValueError(
                'Extrema window length does not match pattern length')
        result = all(a == b
                     if b is not None else a != Extremum.Concavity.INFLECT
                     for a, b in zip(map(lambda x: x.concavity, iter(window)),
                                     self.pattern))
        return result


_inflection_point_pattern = Pattern([
    None,
    Extremum.Concavity.INFLECT,
    Extremum.Concavity.INFLECT,
    Extremum.Concavity.INFLECT,
    None
])


_steer_pattern = Pattern([
    Extremum.Concavity.INFLECT,
    Extremum.Concavity.MINIMUM,
    Extremum.Concavity.INFLECT,
    Extremum.Concavity.MAXIMUM,
    Extremum.Concavity.INFLECT,
    Extremum.Concavity.MINIMUM,
    Extremum.Concavity.INFLECT
])


class SteerEvent(object):
    pattern = _steer_pattern

    def __init__(self, window, trial, steer_angle=None):
        if not self.pattern.match(window):
            raise ValueError('Window does not match pattern.')

        if steer_angle is None:
            steer_angle = trial.filtered_steer_angle()

        self.start = window[0].index
        self.end = window[-1].index
        self.score = self.score(window, steer_angle)
        self.speed = (trial.data.speed[window[0].index:window[-1].index].sum() /
                      (window[-1].index - window[0].index))

    @staticmethod
    def score(window, signal):
        """Calculate the integral of the signal in segments. Inflection points
        are used to split the signal into segments. The integral of each segment
        is multiplied by the minimum or maximum value in that segment.

        If maxima are negative or minima are positive, zero is returned.
        """
        w = window

        if (signal[w[1].index] > 0 or
            signal[w[3].index] < 0 or
            signal[w[5].index] > 0):
            return 0

        score = (signal[w[1].index] * signal[w[0].index:w[2].index].sum() +
                 signal[w[3].index] * signal[w[2].index:w[4].index].sum() +
                 signal[w[5].index] * signal[w[4].index:w[6].index].sum())
        return score
