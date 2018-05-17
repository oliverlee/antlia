# -*- coding: utf-8 -*-
from enum import Enum
from collections import namedtuple

import numpy as np

from antlia.trial import Trial
import antlia.util as util

class EventType(Enum):
    Braking = 0
    Overtaking = 1

    def __str__(self):
        return self.name

EventClassificationData = namedtuple(
        'EventClassificationData',
        ['v0', 'v1', 'vratio'])
EventDetectionData = namedtuple(
        'EventDetectionData',
        ['mask_a', 'mask_b', 'mask_event', 'entry_time', 'exit_time'])

class Trial2(Trial):
    def __init__(self, bicycle_data, lidar_data, period, lidar_bbmask=None):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            # self.event_indices
            # self.event_timerange
            # self.event_detection
            self._detect_event(bbmask_kw=lidar_bbmask)

            # self.event_type
            # self.event_classification
            self._classify_event()

    def _detect_event(self, bbmask_kw=None):
        """Event is detected using two masks, one on the bicycle speed sensor
        and one on the lidar data. Mask A detects when the bicycle speed is
        greater than 0.5. Mask B detects if any object is visible to the lidar,
        within the bounding box specified by (0, 0) and (60, 4) with respect to
        the lidar reference frame.

        This region is then further reduced by detecting the cyclist appearing
        in the bounding box (30, 2), (50, 4) and (possibly, in the case of
        overtaking) and in the bounding box (-20, 2), (-10, 4).

        Parameters:
        bbmask_kw: dict, keywords supplied to lidar.cartesian() for an area to
                   ignore for event detection. This is used in the event of
                   erroneous lidar data.
        """
        mask_a = self.bicycle.speed > 0.5

        bbplus = self.lidar.cartesian(
                    xlim=(0, 60),
                    ylim=(0, 4))[0].count(axis=1)
        if bbmask_kw is not None:
            bbminus = self.lidar.cartesian(**bbmask_kw)[0].count(axis=1)
            mask_b = bbplus - bbminus > 1
        else:
            mask_b = bbplus > 1


        # interpolate mask_b from lidar time to bicycle time
        mask_b = np.interp(self.bicycle.time, self.lidar.time, mask_b)

        events = util.debounce(np.logical_and(mask_a, mask_b))

        edges = np.diff(events.astype(int))
        rising = np.where(edges > 0)[0]
        falling = np.where(edges < 0)[0]
        assert len(rising) == len(falling)

        evti = list(zip(rising, falling))

        # filter out events with a minimum size and minimum average speed
        MIN_TIME_DURATION = 100 # in samples
        MIN_AVG_SPEED = 2 # in m/s
        evti = [e for e in evti
                if ((e[1] - e[0] > MIN_TIME_DURATION) and
                    (np.mean(self.bicycle.speed[e[0]:e[1]]) > MIN_AVG_SPEED))]

        assert len(evti) > 0, "unable to detect event for this trial"
        event = evti[-1]

        # reduce region using entry and exit bounding box detection
        entry_mask = self.lidar.cartesian(
                        xlim=(30, 50),
                        ylim=(2, 4))[0].count(axis=1) > 1
        exit_mask = self.lidar.cartesian(
                        xlim=(-20, -10),
                        ylim=(2, 4))[0].count(axis=1) > 1

        # find time where cyclist enters lidar vision
        entry_time = None
        for x in np.where(entry_mask > 0)[0]:
            t = self.lidar.time[x]
            if (t >= self.bicycle.time[event[0]] and
                t < self.bicycle.time[event[1]]):
                entry_time = t
                try:
                    i = np.where(self.bicycle.time >= t)[0][0]
                except IndexError:
                    # error going from lidar time to bicycle time
                    msg = 'Unable to detect cyclist entry for trial. '
                    msg += 'Event detection failure.'
                    raise IndexError(msg)
                event = (i, event[1])
                break
        assert entry_time is not None, 'No cyclist entry detected'

        # find time where cyclist has finished overtaking obstacle, if it exists
        exit_time = None
        for x in np.where(exit_mask > 0)[0][::-1]:
            t = self.lidar.time[x]
            if (t >= self.bicycle.time[event[0]] and
                t < self.bicycle.time[event[1]]):
                exit_time = t
                try:
                    i = np.where(self.bicycle.time > t)[0][0]
                except IndexError:
                    # error going from lidar time to bicycle time
                    msg = 'Unable to detect cyclist exit for trial. '
                    msg += 'Event detection failure.'
                    raise IndexError(msg)
                event = (event[0], i - 1)
                break

        self.event_detection = EventDetectionData(
                mask_a, mask_b, evti[-1], entry_time, exit_time)
        self.event_indices = event
        self.event_timerange = (self.bicycle.time[event[0]],
                                self.bicycle.time[event[1]])

    def _classify_event(self):
        """Classifies an event as braking or overtaking maneuver.

        The event is classified by comparing the velocity difference at the
        event start and end.
        """
        BRAKING_VELOCITY_THRESHOLD = 0.80

        # Compare first eighth and last eighth
        i0, i1 = self.event_indices
        a0 = i0
        a1 = int(a0 + (i1 - i0)/8)
        b1 = i1
        b0 = int(b1 - (i1 - i0)/8)

        v0 = np.mean(self.bicycle.speed[a0:a1])
        v1 = np.mean(self.bicycle.speed[b0:b1])
        vratio = v1/v0
        self.event_classification = EventClassificationData(v0, v1, vratio)

        #if vratio < BRAKING_VELOCITY_THRESHOLD:
        if self.event_detection.exit_time is None:
            self.event_type = EventType.Braking
        else:
            self.event_type = EventType.Overtaking
