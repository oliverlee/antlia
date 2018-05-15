# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

from antlia.filter import fft
from antlia.pattern import ExtremaList, SteerEvent, window
from antlia.trial import Trial


class Trial2(Trial):
    def __init__(self, bicycle_data, lidar_data, period):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            self._detect_event()

    def _detect_event(self):
        # TODO
        pass

