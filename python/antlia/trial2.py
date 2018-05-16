# -*- coding: utf-8 -*-
import numpy as np

from antlia.trial import Trial
import antlia.util as util


class Trial2(Trial):
    def __init__(self, bicycle_data, lidar_data, period):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            self.event_indices = self._detect_event()
            self.event_timerange = (self.bicycle.time[self.event_indices[0]],
                                    self.bicycle.time[self.event_indices[1]])

    def _detect_event(self):
        mask_a = self.bicycle.speed > 1
        mask_b = self.lidar.cartesian(
                    xlim=(0, 60),
                    ylim=(0, 4))[0].count(axis=1) > 1

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
