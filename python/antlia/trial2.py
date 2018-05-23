# -*- coding: utf-8 -*-
from enum import Enum
from collections import namedtuple
import warnings

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
import hdbscan

import antlia.filter as ff
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

ClusterData = namedtuple(
        'ClusterData',
        ['label', 'index', 'zmean', 'zspan', 'stationary'])

ENTRY_BB = {
    'xlim': (20, 50),
    'ylim': (2, 3.5)
}
EXIT_BB = {
    'xlim': (-20, -10),
    'ylim': (2, 3.5)
}
OBSTACLE_BB = {
    'xlim': (-5, -2),
    'ylim': (2.90, 3.25)
}
VALID_BB = {
    'xlim': (-20, 60),
    'ylim': (1.0, 3.5)
}


class Event(Trial):
    def __init__(self, bicycle_data, lidar_data, period, event_type):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data
            self.type = event_type

            self.x = None
            self.y = None
            self.z = None
            self.valid_points = None
            self.hdb = None
            self.clusters = None
            self.bb_mask = None
            self.stationary_mask = None
            self.stationary_count = None
            self._identify_stationary()

    def _identify_stationary(self, min_zspan=0.5, zscale=0.0005, hdbscan_kw=None):
        x, y, z = self.lidar.cartesianz(**VALID_BB)
        bb_mask = x.mask

        # rescale z
        assert z.shape[0] > 1
        z.mask = False
        z -= z[0, 0]
        z *= zscale/(z[1, 0] - z[0, 0])
        z.mask = bb_mask

        # create point cloud data
        X = np.vstack((
            x.compressed(),
            y.compressed(),
            z.compressed())).transpose()

        if hdbscan_kw is None:
            hdbscan_kw = {}

        hdbscan_kw['allow_single_cluster'] = False
        hdbscan_kw.setdefault('min_cluster_size', 30)
        hdbscan_kw.setdefault('min_samples', 15)
        hdbscan_kw.setdefault('metric', 'euclidean')

        # cluster
        hdb = hdbscan.HDBSCAN(**hdbscan_kw).fit(X)
        cluster_labels = list(set(hdb.labels_))

        # determine cluster data and stationarity
        bb_mask_shape = bb_mask.shape
        bb_mask = bb_mask.reshape(-1)
        not_bb_mask_index = np.where(~bb_mask)[0]
        stationary_mask = np.zeros(bb_mask.shape, dtype=bool)
        cluster_data = []
        stationary_count = 0
        for label in cluster_labels:
            index = hdb.labels_ == label

            zmean = X[index, 2].mean()
            zspan = len(set(X[index, 2]))

            # (non-noise) clusters with large zspan
            stationary = label != -1 and zspan > min_zspan*z.shape[0]

            cluster_data.append(ClusterData(
                label, index, zmean, zspan, stationary))

            if stationary:
                stationary_count += 1
                stationary_mask[not_bb_mask_index[index]] = True

        bb_mask = bb_mask.reshape(bb_mask_shape)
        stationary_mask = stationary_mask.reshape(bb_mask_shape)

        self.x = x
        self.y = y
        self.z = z
        self.valid_points = X
        self.hdb = hdb
        self.clusters = cluster_data
        self.bb_mask = bb_mask
        self.stationary_mask = stationary_mask
        self.stationary_count = stationary_count

    def plot_clusters(self, plot_cluster_func=None, ax=None, **fig_kw):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, **fig_kw)
        else:
            fig = ax.get_figure()

        noise_color = 'dimgray'
        stationary_colors = sns.husl_palette(self.stationary_count, l=0.3)
        colors = sns.husl_palette(
                len(self.clusters) - self.stationary_count - 1,
                s=0.5)

        for cluster in self.clusters:
            X = self.valid_points[cluster.index]

            if plot_cluster_func is not None:
                plot_kw = plot_cluster_func(cluster)
                plot_kw.setdefault('marker', '.')
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], **plot_kw)
            else:
                if cluster.label == -1:
                    color = noise_color
                    alpha = 0.2
                else:
                    if cluster.stationary:
                        color = stationary_colors.pop()
                    else:
                        color = colors.pop()
                    alpha = 1
                ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                           marker='.', color=color, alpha=alpha)


        return fig, ax

    def plot_trajectory(self, ax=None, **fig_kw):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=False, **fig_kw)
        else:
            assert len(ax) == 2
            fig = ax[0].get_figure()

        colors = sns.color_palette('Paired', 12)

        # stationary points
        ax[0].scatter(self.x.data[self.stationary_mask & ~self.bb_mask],
                      self.y.data[self.stationary_mask & ~self.bb_mask],
                      s=5, marker='x', color='black',
                      label='stationary points')

        x = self.x.copy()
        y = self.y.copy()
        x.mask = self.stationary_mask | self.bb_mask
        y.mask = self.stationary_mask | self.bb_mask

        # non-stationary points
        ax[0].scatter(x, y, s=3, marker='.', color=colors[1],
                      label='non-stationary points')

        # trajectory points
        xm = x.mean(axis=1)
        ym = y.mean(axis=1)
        ax[0].scatter(xm, ym, s=5, edgecolor=colors[5],
                      label='NSP centroid (per frame)')

        # interpolated trajectory
        xm[xm.mask] = np.interp(
                np.where(xm.mask)[0], np.where(~xm.mask)[0], xm[~xm.mask])
        ym[ym.mask] = np.interp(
                np.where(ym.mask)[0], np.where(~ym.mask)[0], ym[~ym.mask])
        ax[0].plot(xm, ym, color=colors[4],
                   label='NSP centroid (interpolated)')

        # filtered trajectory
        order = 4
        fc = 1.5
        fs = 20
        wn = fc / (0.5*fs)
        b, a = scipy.signal.butter(order, wn, btype='lowpass')
        butterf = lambda x: scipy.signal.filtfilt(b, a, x)
        ax[0].plot(butterf(xm), butterf(ym), color=colors[3],
                   label='NSP centroid (filtered, low pass butterworth)')

        handles, labels = ax[0].get_legend_handles_labels()
        # move first 2 elements to end
        handles = handles[2:] + handles[:2]
        lables = labels[2:] + labels[:2]
        ax[0].legend(handles, labels)

        f = lambda x: np.square(np.diff(x))
        v = lambda x, y: np.sqrt(f(x) + f(y)) / 0.05
        ax[1].plot(self.bicycle.time,
                   ff.moving_average(self.bicycle.speed, 55),
                   color=colors[1], zorder=2,
                   label='measured speed (filtered, moving average)')
        ylim = ax[1].get_ylim()
        ax[1].plot(self.bicycle.time,
                   self.bicycle.speed,
                   color=colors[0], zorder=0,
                   label='measured speed')
        ax[1].plot(self.lidar.time[1:], v(xm, ym),
                   color=colors[4], zorder=1,
                   label='estimated speed (centroid)')
        ax[1].plot(self.lidar.time[1:], v(butterf(xm), butterf(ym)),
                   color=colors[3], zorder=2,
                   label='estimated speed (centroid, filtered, low pass butter)')
        ax[1].set_ylim(ylim)

        handles, labels = ax[1].get_legend_handles_labels()
        # swap first two elements
        handles[0], handles[1] = handles[1], handles[0]
        labels[0], labels[1] = labels[1], labels[0]
        ax[1].legend(handles, labels)

        return fig, ax



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
        bbplus = lidar_data.cartesian(**VALID_BB)[0].count(axis=1)

        # subtract the obstacle
        bbplus -= lidar_data.cartesian(**OBSTACLE_BB)[0].count(axis=1)

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
        entry_mask = self.lidar.cartesian(**ENTRY_BB)[0].count(axis=1) > 1
        exit_mask = self.lidar.cartesian(**EXIT_BB)[0].count(axis=1) > 1

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
