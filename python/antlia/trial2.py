# -*- coding: utf-8 -*-
from enum import Enum

import numpy as np

from antlia.trial import Trial
import antlia.util as util

class EventType(Enum):
    Braking = 0
    Overtaking = 1

    def __str__(self):
        return self.name


class Trial2(Trial):
    def __init__(self, bicycle_data, lidar_data, period):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            self.event_indices = self._detect_event()
            self.event_timerange = (self.bicycle.time[self.event_indices[0]],
                                    self.bicycle.time[self.event_indices[1]])

            self.event_type = self._classify_event()

    def _detect_event(self):
        """Event is detected using two masks, one on the bicycle speed sensor
        and one on the lidar data. Mask A detects when the bicycle speed is
        greater than 1. Mask B detects if any object is visible to the lidar,
        within the bounding box specified by (0, 0) and (60, 4) with respect to
        the lidar reference frame.

        A small negative bounding box is also used to ignore errorneous lidar
        data.
        """
        mask_a = self.bicycle.speed > 1

        bbplus = self.lidar.cartesian(
                    xlim=(0, 60),
                    ylim=(0, 4))[0].count(axis=1)
        bbminus = self.lidar.cartesian(
                    xlim=(3.8, 4.3),
                    ylim=(2.9, 3.4))[0].count(axis=1)
        bbplus -= bbminus

        mask_b = bbplus > 1

        # interpolate mask_b from lidar time to bicycle time
        mask_b = np.interp(self.bicycle.time, self.lidar.time, mask_b)

        events = util.debounce(np.logical_and(mask_a, mask_b))

        edges = np.diff(events.astype(int))
        rising = np.where(edges > 0)[0]
        falling = np.where(edges < 0)[0]
        assert len(rising) == len(falling)

        evti = list(zip(rising, falling))

        # filter out events with a minimum size
        evti = [e for e in evti if e[1] - e[0] > 100]

        assert len(evti) > 0, "unable to detect event for this trial"

        event = evti[-1]
        return evti[-1]

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
        if v1/v0 < BRAKING_VELOCITY_THRESHOLD:
            return EventType.Braking

        return EventType.Overtaking

