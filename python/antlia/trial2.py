# -*- coding: utf-8 -*-
from enum import Enum
from collections import namedtuple
import warnings

import numpy as np

from antlia.trial import Trial
import antlia.util as util

class EventType(Enum):
    Braking = 0
    Overtaking = 1

    def __str__(self):
        return self.name

EventDetectionData = namedtuple(
        'EventDetectionData',
        ['mask_a', 'mask_b', 'mask_event', 'entry_time', 'exit_time'])


class Event(Trial):
    def __init__(self, bicycle_data, lidar_data, period, event_type):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data
            self.type = event_type


class Trial2(Trial):
    def __init__(self, bicycle_data, lidar_data, period, lidar_bbmask=None):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            self.event_indices = None
            self.event_detection = None
            self.event = None

            self._detect_event(bbmask_kw=lidar_bbmask)

    @staticmethod
    def mask_a(bicycle_data):
        return bicycle_data.speed > 0.5

    @staticmethod
    def mask_b(lidar_data, bbmask_kw=None):
        bbplus = lidar_data.cartesian(
                    xlim=(-20, 60),
                    ylim=(1.0, 3.5))[0].count(axis=1)

        # subtract the obstacle
        bbplus -= lidar_data.cartesian(
                    xlim=(-5, -2),
                    ylim=(2.90, 3.25))[0].count(axis=1)

        if bbmask_kw is not None:
            bbminus = lidar_data.cartesian(**bbmask_kw)[0].count(axis=1)
            mask_b = bbplus - bbminus > 1
        else:
            mask_b = bbplus > 1
        return mask_b

    @staticmethod
    def event_indices(mask):
        edges = np.diff(mask.astype(int))
        rising = np.where(edges > 0)[0]
        falling = np.where(edges < 0)[0]

        assert len(rising) == len(falling)
        return list(zip(rising, falling))

    def _detect_event(self, bbmask_kw=None):
        """Event is detected using two masks, one on the bicycle speed sensor
        and one on the lidar data. Mask A detects when the bicycle speed is
        greater than 0.5. Mask B detects if any object is visible to the lidar,
        within the bounding box specified by (-20, 1.0) and (60, 3.5) with
        respect to the lidar reference frame. The obstacle is ignored with the
        use of a negative bounding box.

        This region is then further reduced by detecting the cyclist appearing
        in the bounding box (20, 2), (50, 3.5) and (possibly, in the case of
        overtaking) and in the bounding box (-20, 2), (-10, 3.5).

        Parameters:
        bbmask_kw: dict, keywords supplied to lidar.cartesian() for an area to
                   ignore for event detection. This is used in the event of
                   erroneous lidar data.
        """
        mask_a = Trial2.mask_a(self.bicycle)
        mask_b = Trial2.mask_b(self.lidar, bbmask_kw)

        # interpolate mask_b from lidar time to bicycle time
        mask_b = np.interp(self.bicycle.time, self.lidar.time, mask_b)

        mask_ab = util.debounce(np.logical_and(mask_a, mask_b))
        evti = Trial2.event_indices(mask_ab)

        # filter out events with a minimum size and minimum average speed
        MIN_TIME_DURATION = int(5.5 * 125) # in samples
        MIN_AVG_SPEED = 2 # in m/s
        evti = [e for e in evti
                if ((e[1] - e[0] > MIN_TIME_DURATION) and
                    (np.mean(self.bicycle.speed[e[0]:e[1]]) > MIN_AVG_SPEED))]

        assert len(evti) > 0, "unable to detect event for this trial"
        evt_index = evti[-1]

        # reduce region using entry and exit bounding box detection
        entry_mask = self.lidar.cartesian(
                        xlim=(20, 50),
                        ylim=(2, 3.5))[0].count(axis=1) > 1
        exit_mask = self.lidar.cartesian(
                        xlim=(-20, -10),
                        ylim=(2, 3.5))[0].count(axis=1) > 1

        # find time where cyclist enters lidar vision
        entry_time = None
        for x in np.where(entry_mask > 0)[0]:
            t = self.lidar.time[x]
            if (t >= self.bicycle.time[evt_index[0]] and
                t < self.bicycle.time[evt_index[1]]):
                entry_time = t
                try:
                    i = np.where(self.bicycle.time >= t)[0][0]
                except IndexError:
                    # error going from lidar time to bicycle time
                    msg = 'Unable to detect cyclist entry for trial. '
                    msg += 'Event detection failure.'
                    raise IndexError(msg)
                evt_index = (i, evt_index[1])
                break
        if entry_time is None:
            msg = 'Unable to detect cyclist entry for event starting at '
            msg += 't = {0:.3f} seconds'.format(self.bicycle.time[evt_index[0]])
            warnings.warn(msg, UserWarning)

        # find time where cyclist has finished overtaking obstacle, if it exists
        exit_time = None
        for x in np.where(exit_mask > 0)[0][::-1]:
            t = self.lidar.time[x]
            if (t >= self.bicycle.time[evt_index[0]] and
                t < self.bicycle.time[evt_index[1]]):
                exit_time = t
                try:
                    i = np.where(self.bicycle.time > t)[0][0]
                except IndexError:
                    # error going from lidar time to bicycle time
                    msg = 'Unable to detect cyclist exit for trial. '
                    msg += 'Event detection failure.'
                    raise IndexError(msg)
                evt_index = (evt_index[0], i - 1)
                break

        d = (evt_index[1] - evt_index[0])//8
        v0 = np.mean(self.bicycle.speed[evt_index[0]:evt_index[0] + d])
        v1 = np.mean(self.bicycle.speed[evt_index[1] - d:evt_index[1]])
        if exit_time is None and v1/v0 > 0.8:
            msg = 'Unable to detect cyclist exiting or braking for event '
            msg += 'ending at t = '
            msg += '{0:.3f} seconds'.format(self.bicycle.time[evt_index[1]])
            warnings.warn(msg, UserWarning)

        # classify event type
        event_type = EventType.Overtaking
        if exit_time is None:
            event_type = EventType.Braking

        self.event_indices = evt_index
        self.event_detection = EventDetectionData(
                mask_a, mask_b, evti[-1], entry_time, exit_time)

        t0 = self.bicycle.time[evt_index[0]]
        t1 = self.bicycle.time[evt_index[1]] + 0.05 # add one extra lidar frame
        self.event = Event(
                self.bicycle[evt_index[0]:evt_index[1]],
                self.lidar.frame(lambda t: (t >= t0) & (t < t1)),
                self.period,
                event_type)
