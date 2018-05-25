# -*- coding: utf-8 -*-
from enum import Enum
from collections import namedtuple
import warnings

import numpy as np
import scipy.signal
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
import hdbscan

from antlia import dtc
import antlia.filter as ff
from antlia.trial import Trial
import antlia.util as util

EventDetectionData = namedtuple(
        'EventDetectionData',
        ['mask_a', 'mask_b', 'mask_event', 'entry_time', 'exit_time'])

ClusterData = namedtuple(
        'ClusterData',
        ['label', 'index', 'zmean', 'zspan', 'count', 'area', 'stationary'])

FakeHdb = namedtuple(
        'FakeHdb',
        ['labels_'])

ENTRY_BB = {
    'xlim': (20, 30),
    'ylim': (2.5, 3.5)
}
EXIT_BB = {
    'xlim': (-20, -10),
    'ylim': (2.5, 3.5)
}
OBSTACLE_BB = {
    'xlim': (-5, -2),
    'ylim': (2.70, 3.25)
}
VALID_BB = {
    'xlim': (-20, 60),
    'ylim': (1.0, 3.5)
}


class EventType(Enum):
    Braking = 0
    Overtaking = 1

    def __str__(self):
        return self.name


class Event(Trial):
    def __init__(self, bicycle_data, lidar_data, period, event_type,
                 invalid_bb=None):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data
            self.type = event_type
            self.invalid_bb = invalid_bb

            self.x = None
            self.y = None
            self.z = None
            self.valid_points = None
            self.hdb = None
            self.clusters = None
            self.bb_mask = None
            self.stationary_mask = None
            self.stationary_count = None
            self._identify_stationary(invalid_bb=invalid_bb)

    def _identify_stationary(self, min_zspan=0.7, zscale=0.0005,
                             hdbscan_kw=None, invalid_bb=None):
        x, y, z = self.lidar.cartesianz(**VALID_BB)

        # exclusive bbmask
        if invalid_bb is not None:
            if not hasattr(invalid_bb, '__iter__'):
                invalid_bb = [invalid_bb]

            for m in invalid_bb:
                if 'xlim' in m and 'ylim' in m:
                    xmin, xmax = m['xlim']
                    ymin, ymax = m['ylim']
                    index = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
                    x[index] = ma.masked

        # use the same mask for x and y
        y.mask = x.mask

        # rescale z and then use the same mask as x and y
        assert z.shape[0] > 1
        z.mask = False
        z -= z[0, 0]
        z *= zscale/(z[1, 0] - z[0, 0])
        z.mask = x.mask

        # make sure (at least) some entries in x, y, z are masked so we don't
        # perform clustering on all the points at the lidar boundary
        assert np.any(x.mask)
        assert np.any(y.mask)
        assert np.any(z.mask)

        # copy x.mask so we aren't referencing the same data
        bb_mask = np.ma.getmaskarray(x).copy()

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
        bb_mask = bb_mask.reshape(-1)
        not_bb_mask_index = np.where(~bb_mask)[0]
        stationary_mask = np.zeros(bb_mask.shape, dtype=bool)
        cluster_data = []
        stationary_count = 0

        zrange = X[-1, 2] - X[0, 2]
        zmidpoint = zrange/2

        indexA = X[:, 2] < zrange/4
        indexB = (X[:, 2] >= zrange/4) & (X[:, 2] <= 2*zrange/4)
        indexC = (X[:, 2] >= 2*zrange/4) & (X[:, 2] <= 3*zrange/4)
        indexD = X[:, 2] >= 3*zrange/4
        indexi = [indexA, indexB, indexC, indexD]

        area_limit = 0.2
        area = lambda x: x[:, :2].ptp(axis=0).prod()
        zmean = lambda i: X[i, 2].mean()
        zspan = lambda i: len(set(X[i, 2]))

        outside_midpoint = lambda i: abs(zmean(i) - zmidpoint) > 0.1*zrange
        large_area = lambda x: area(x) > area_limit

        extra_cluster_index = np.zeros(hdb.labels_.shape, dtype=bool)
        for label in cluster_labels:
            index = hdb.labels_ == label

            # (non-noise) clusters with large zspan
            stationary = label != -1 and zspan(index) > min_zspan*z.shape[0]

            # however if zmean is not near zmidpoint OR the xy area is large
            # part of the cyclist trajectory has been grouped into this cluster
            # and we must manually split it
            if (stationary and
                (outside_midpoint(index) or large_area(X[index]))):

                # determine which set has a smaller xy area/bounding box
                Xj = None
                min_area = None
                for i in indexi:
                    Xi = X[i][index[i]]

                    if Xi.shape[0] < 10:
                        # skip index set with fewer cluster points than minimum
                        continue

                    a = area(Xi)
                    if min_area is None or a < min_area:
                        min_area = a
                        Xj = Xi

                # get bounding box for half where the cyclist is _not_ there
                xmin, ymin, _ = Xj.min(axis=0)
                xmax, ymax, _ = Xj.max(axis=0)

                # track indices with a masked array
                Y = np.ma.masked_array(X)
                Y[~index] = np.ma.masked

                # get all points within bounding box for both halves
                within = ((Y[:, 0] > xmin) & (Y[:, 0] < xmax) &
                          (Y[:, 1] > ymin) & (Y[:, 1] < ymax))
                index = within

                # determine points associated with cyclist
                Y[within] = np.ma.masked
                extra_cluster_index |= ~Y.mask[:, 0]

            cluster_data.append(ClusterData(
                label,
                index,
                zmean(index),
                zspan(index),
                np.count_nonzero(index),
                area(X[index]),
                stationary))

            if stationary:
                stationary_count += 1
                stationary_mask[not_bb_mask_index[index]] = True

        # if we manually split a cluster, need to add new cluster containing
        # all the excluded points
        if np.any(extra_cluster_index):
            index = extra_cluster_index
            cluster_data.append(ClusterData(
                max(cluster_labels) + 1,
                index,
                zmean(index),
                zspan(index),
                np.count_nonzero(index),
                area(X[index]),
                False))

        bb_mask = bb_mask.reshape(x.mask.shape)
        stationary_mask = stationary_mask.reshape(x.mask.shape)

        self.x = x
        self.y = y
        self.z = z
        self.valid_points = X
        self.hdb = hdb
        self.clusters = cluster_data
        self.bb_mask = bb_mask
        self.stationary_mask = stationary_mask
        self.stationary_count = stationary_count

    @staticmethod
    def _set_radius_mask(X, radius=1.5):
        """Sets the mask of X to invalidate points far from the cluster
        centroid, where X contains the points for a single lidar frame.

        Parameters:
        X: masked array with shape (n, 2)
        """
        if len(X.shape) != 2:
            raise TypeError('X.shape must be (n, 2)')
        if X.shape[1] != 2:
            raise TypeError('X.shape must be (n, 2)')

        X.mask = np.ma.nomask
        #centroid = X.mean(axis=0)
        centroid = np.median(X.data, axis=0)
        distances = scipy.spatial.distance.cdist(X, centroid.reshape((1, 2)))
        index = np.matlib.repmat(
                    distances > radius,
                    1,
                    2)
        X[index] = np.ma.masked

    @staticmethod
    def _set_obstacle_mask(Y):
        """Sets the mask of Y to exclude stationary noise, where Y contains the
        points for a single lidar frame.

        Parameters:
        Y: masked array with shape (n, 2)
        """
        if len(Y.shape) != 2:
            raise TypeError('Y.shape must be (n, 2)')
        if Y.shape[1] != 2:
            raise TypeError('Y.shape must be (n, 2)')

        Y.mask = np.ma.nomask

        hdbscan_kw = {}
        hdbscan_kw['allow_single_cluster'] = True
        hdbscan_kw['min_cluster_size'] = 30
        hdbscan_kw['min_samples'] = 15
        hdbscan_kw['metric'] = 'euclidean'

        # try clustering to find obstacle
        try:
            hdb, labels = Event.__get_single_cluster(Y, hdbscan_kw)
        except ValueError:
            hdbscan_kw['min_cluster_size'] = 15
            hdbscan_kw['min_samples'] = 10
            hdb, labels = Event.__get_single_cluster(Y, hdbscan_kw)

        Y[hdb.labels_ != 0] = np.ma.masked

    def _compressed_points(self, lidar_index, stationary):
        """Return an array with the points at a given lidar frame.

        Parameters:
        lidar_index: int, valid lidar frame index within the Event
        stationary: bool, if the stationary_mask should be applied
        """
        if stationary:
            mask = self.bb_mask | self.stationary_mask
        else:
            mask = self.bb_mask | ~self.stationary_mask
        x = np.ma.masked_where(mask, self.x, copy=True)
        y = np.ma.masked_where(mask, self.y, copy=True)

        X = np.ma.masked_array(np.vstack((
                x[lidar_index].compressed(),
                y[lidar_index].compressed())).T)
        return X

    def _calculate_dtc(self, lidar_index):
        """Calculate the distance between the bicycle and obstacle at lidar
        index. Returns the calculated distance, bicycle points, and stationary
        points.
        """
        # bicycle
        X = self._compressed_points(lidar_index, False)

        # stationary
        Y = self._compressed_points(lidar_index, True)

        # exclude bicycle noise
        self._set_radius_mask(X)

        # exclude stationary noise
        #self._set_obstacle_mask(Y)
        self._set_radius_mask(Y)

        # get closest pair and perform distance calculation
        pair = dtc.bcp(X, Y)
        dist = dtc.dist(*pair)
        return dist, pair, X, Y

    def _plot_closest_pair(self, lidar_index, ax=None, **fig_kw):
        _, pair, X, Y = self._calculate_dtc(lidar_index)

        if ax is None:
            fig, ax = plt.subplots(**fig_kw)
        else:
            fig = ax.get_figure()

        colors = sns.color_palette('Paired', 10)
        dtc.plot_closest_pair(X, Y, pair=pair, ax=ax, color=colors[1::2])

        # plot bicycle noise
        X.mask = ~X.mask
        ax.scatter(*X.T, color=colors[0])

        # plot stationary noise
        Y.mask = ~Y.mask
        ax.scatter(*Y.T, color=colors[2])

        return fig, ax

    def calculate_dtc(self, timestamp):
        """Return the distance between the bicycle and obstacle at timestamp.
        """
        assert timestamp >= self.bicycle.time[0]
        assert timestamp <= self.bicycle.time[-1]
        assert timestamp >= self.lidar.time[0]
        assert timestamp <= self.lidar.time[-1]

        i = self.lidar.frame_index(timestamp)
        assert i > 0

        times = []
        distances = []
        for j in [i - 1, i]:
            times.append(self.lidar.time[j][0])
            distances.append(self._calculate_dtc(j)[0])

        return np.interp(timestamp, times, distances)

    @staticmethod
    def __get_single_cluster(X, hdbscan_kw, raise_error=True):
        # if all points are within obstacle bounding box, skip clustering
        if (np.all(X[:, 0] > OBSTACLE_BB['xlim'][0]) and
            np.all(X[:, 0] < OBSTACLE_BB['xlim'][1]) and
            np.all(X[:, 1] > OBSTACLE_BB['ylim'][0]) and
            np.all(X[:, 1] < OBSTACLE_BB['ylim'][1])):

            # we only use the 'labels_' attribute of hdb
            hdb = FakeHdb(np.zeros(X.shape[0],))
            labels = [0]
            return hdb, labels

        hdb = hdbscan.HDBSCAN(**hdbscan_kw).fit(X)
        labels = list(set(hdb.labels_))

        if raise_error:
            Event.__check_single_cluster(hdb, labels, hdbscan_kw)

        return hdb, labels

    @staticmethod
    def __check_single_cluster(hdb, labels, hdbscan_kw, warn=False):
        if max(labels) != 0:
            msg = 'Found more than one cluster: {}'.format(labels)
            msg += '\nChange cluster parameters: {}'.format(hdbscan_kw)
            if warn:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

        noise_count = np.count_nonzero(hdb.labels_ == -1)
        n = len(hdb.labels_)
        if noise_count > 0.2*n:
            msg = 'More than 20% of points are classified as noise:'
            msg += ' ({}/{})'.format(noise_count, n)
            msg += '\nChange cluster parameters: {}'.format(hdbscan_kw)
            if warn:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

    def _plot_stationary_clusters(self, lidar_index,
                                  hdbscan_kw=None, ax=None, **fig_kw):
        """Plot stationary clusters for a single lidar frame. """
        # stationary
        X = self._compressed_points(lidar_index, True)

        if hdbscan_kw is None:
            hdbscan_kw = {
                'min_cluster_size': 30,
                'min_samples': 15,
                'metric': 'euclidean',
                'allow_single_cluster': True,
            }

            # try clustering to find obstacle
            try:
                hdb, labels = self.__get_single_cluster(X, hdbscan_kw)
            except ValueError:
                hdbscan_kw['min_cluster_size'] = 15
                hdbscan_kw['min_samples'] = 10
                hdb, labels = self.__get_single_cluster(X, hdbscan_kw,
                                                        raise_error=False)
        else:
            hdb, labels = self.__get_single_cluster(X, hdbscan_kw,
                                                    raise_error=False)
        print('Using {}'.format(hdbscan_kw))
        self.__check_single_cluster(hdb, labels, hdbscan_kw, warn=True)

        noise_color = 'dimgray'
        n_colors = len(labels) - 1
        if n_colors <= 10:
            colors = sns.color_palette('tab10', 10)
        else:
            colors = sns.husl_palette(n_colors, l=0.7)

        if ax is None:
            fig, ax = plt.subplots(**fig_kw)
        else:
            fig = ax.get_figure()

        for i in labels:
            index = hdb.labels_ == i
            if i == -1:
                color = noise_color
                label = 'stationary noise'
            else:
                color = colors[i]
                label = 'obstacle'
            ax.scatter(*X[index].T, color=color, label=label)
        ax.legend()

        return fig, ax

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
    def __init__(self, bicycle_data, lidar_data, period, invalid_bb=None):
            super().__init__(bicycle_data, period)
            self.lidar = lidar_data
            self.bicycle = self.data

            self.event_indices = None
            self.event_detection = None
            self.event = None
            self._detect_event(invalid_bb=invalid_bb)

    @staticmethod
    def mask_a(bicycle_data):
        return bicycle_data.speed > 0.5

    @staticmethod
    def mask_b(lidar_data, invalid_bb=None):
        bbplus = lidar_data.cartesian(**VALID_BB)[0].count(axis=1)

        # subtract the obstacle
        bbplus -= lidar_data.cartesian(**OBSTACLE_BB)[0].count(axis=1)

        mask = bbplus
        if invalid_bb is not None:
            if not hasattr(invalid_bb, '__iter__'):
                invalid_bb = [invalid_bb]

            for bbminus in invalid_bb:
                mask -= lidar_data.cartesian(**bbminus)[0].count(axis=1)
        return mask > 1

    @staticmethod
    def event_indices(mask):
        edges = np.diff(mask.astype(int))
        rising = np.where(edges > 0)[0]
        falling = np.where(edges < 0)[0]

        assert len(rising) == len(falling)
        return list(zip(rising, falling))

    def _detect_event(self, invalid_bb=None):
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
        invalid_bb: dict or iteratble of dicts, keywords supplied to
                    lidar.cartesian() for an area to ignore for event detection.
                    This is used in the event of erroneous lidar data.
        """
        mask_a = Trial2.mask_a(self.bicycle)
        mask_b = Trial2.mask_b(self.lidar, invalid_bb)

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
        # using the falling edge of the first clump
        exit_time = None
        exit_clumps = np.ma.extras._ezclump(exit_mask > 0)
        for clump in exit_clumps:
            t = self.lidar.time[clump.stop - 1] # make exit time inclusive
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
                event_type,
                invalid_bb=invalid_bb)
