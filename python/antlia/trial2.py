# -*- coding: utf-8 -*-
import bisect
import enum
from collections import namedtuple
import heapq
import itertools
import warnings

import numpy as np
import scipy.signal
import scipy.spatial
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
import hdbscan

from antlia import dtc
import antlia.filter as ff
from antlia.trial import Trial
import antlia.util as util
import antlia.plot_braking as braking

EventDetectionData = namedtuple(
        'EventDetectionData',
        ['valid_clumps', 'entry_clumps',
         'exit_brake_clumps', 'exit_steer_clumps',
         'event_slice', 'entry_time', 'exit_time'])

ClusterData = namedtuple(
        'ClusterData',
        ['label', 'index', 'centroid', 'span',
         'count', 'area', 'stationary'])

FakeHdb = namedtuple(
        'FakeHdb',
        ['labels_'])

# Leave this defined to load saved data
SteeringIdentificationCase = namedtuple(
        'SteeringIdentificationCase',
        ['attenuation', 'data', 'freq', 'xf', 'minima', 'maxima', 'inflections',
         'cutoff', 'section', 'score'])

# Leave this defined to load saved data
SteerEventSinusoidFitParameters = namedtuple(
        'SteerEventSinusoidFitParameters',
        ['amplitude',
         'frequency',
         'phase',
         'mean']) # TODO add all information from
                  # steering identification case object?
SteerEventGaussianFitParameters = namedtuple(
        'SteerEventGaussianFitParameters',
        ['index_start',
         'index_apex',
         'index_end',
         'gaussianfit_amplitude',
         'gaussianfit_mean',
         'gaussianfit_std',
         'gaussianfit_offset'])

# braking identification done in a single step
BrakeEventLinearFitParameters = namedtuple(
        'BrakeEventLinearFitParameters',
        ['average_window_size',
         'braking_threshold',
         'slice_minsize',
         'signal_time',
         'filtered_velocity',
         'filtered_acceleration',
         'braking_slice',
         'lockup_mask',
         'linregress_slope',
         'linregress_intercept',
         'linregress_rvalue',
         'linregress_pvalue',
         'linregress_stderr'])

ENTRY_BB = {
    'xlim': (18, 25),
    'ylim': (2.5, 3.5)
}
EXIT_BB_BRAKE = {
    'xlim': (-5, 0),
    'ylim': (0.5, 1.5)
}
EXIT_BB_STEER = {
    'xlim': (-20, -15),
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
OBSTACLE_POINT = (-3.66, 3.06)


# Leave this defined to load saved data
class SteeringIdentification(object):
    def __init__(self, attenuation, pattern):
        self.attenuation = attenuation
        self.pattern = pattern
        self.cases = []

    def add_case(self, attenuation_value, data, freq, xf,
                 minima, maxima, inflections, cutoff, section, score):
        assert np.any(np.isin(attenuation_value, self.attenuation))
        self.cases.append(SteeringIdentificationCase(
            attenuation_value,
            data,
            freq,
            xf,
            minima,
            maxima,
            inflections,
            cutoff,
            section,
            score))


class EventType(enum.Enum):
    Braking = 0
    Overtaking = 1

    def __str__(self):
        return self.name


class Event(Trial):
    def __init__(self, trial, bicycle_data, lidar_data, period, event_type,
                 bbmask=None):
            super().__init__(bicycle_data, period)
            self.trial = trial
            self.lidar = lidar_data
            self.bicycle = self.data
            self.type = event_type
            self.bbmask = bbmask

            self.x = None
            self.y = None
            self.z = None
            self.valid_points = None
            self.hdb = None
            self.clusters = None
            self.bb_mask = None
            self.stationary_mask = None
            self.stationary_count = None
            self._identify_stationary(bbmask=bbmask)

            # Leave this defined to load saved data
            self.si = None
            self.steer_slice = None # bicycle time
            #self._identify_steer_slice() # don't run automatically

            self.kalman_result = None
            self.kalman_smoothed_result = None

    def _identify_stationary(self, min_zspan=0.7, zscale=0.0005,
                             hdbscan_kw=None, bbmask=None):
        x, y, z = self.lidar.cartesianz(**VALID_BB)

        # apply bounding box masks if specified
        if bbmask is not None:
            _apply_bbmask(bbmask, x, y, z)

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
            hdbscan_kw = {
                'min_cluster_size': 30,
                'min_samples': 15,
                'metric': 'euclidean'
            }

        hdbscan_kw['allow_single_cluster'] = False

        # cluster
        hdb = hdbscan.HDBSCAN(**hdbscan_kw).fit(X)
        cluster_labels = list(set(hdb.labels_))

        # change parameters to get fewer clusters
        if len(cluster_labels) > 80:
            hdbscan_kw['min_cluster_size'] = 60
            hdbscan_kw['min_samples'] = 40
            hdb = hdbscan.HDBSCAN(**hdbscan_kw).fit(X)
            cluster_labels = list(set(hdb.labels_))

        # determine cluster data and stationarity
        bb_mask = bb_mask.reshape(-1)
        not_bb_mask_index = np.where(~bb_mask)[0]
        stationary_mask = np.zeros(bb_mask.shape, dtype=bool)
        cluster_data = []
        stationary_count = 0

        zmax = X[-1, 2]
        zmin = X[0, 2]
        zrange = zmax - zmin

        indexA = X[:, 2] < zrange/4
        indexB = (X[:, 2] >= zrange/4) & (X[:, 2] <= 2*zrange/4)
        indexC = (X[:, 2] >= 2*zrange/4) & (X[:, 2] <= 3*zrange/4)
        indexD = X[:, 2] >= 3*zrange/4
        indexi = [indexA, indexB, indexC, indexD]

        area_limit = 0.2
        area = lambda x: x[:, :2].ptp(axis=0).prod()
        # x, y is a normalized coordinates
        zmean = lambda i: X[i, 2].mean()/zrange
        zspan = lambda i: (np.max(X[i, 2]) - np.min(X[i, 2]))/zrange
        # x, y are not normalized coordinates
        xmean = lambda i: X[i, 0].mean()
        ymean = lambda i: X[i, 1].mean()
        xspan = lambda i: np.max(X[i, 0]) - np.min(X[i, 0])
        yspan = lambda i: np.max(X[i, 1]) - np.min(X[i, 1])

        starts_near_zmin = lambda i: np.min(X[i, 2]) < zmin + 0.1*zrange
        ends_near_zmax = lambda i: np.max(X[i, 2]) > zmax - 0.1*zrange

        def within_bb(x, y, bb):
            return (x < bb['xlim'][1] and x > bb['xlim'][0] and
                    y < bb['ylim'][1] and y > bb['ylim'][0])

        extra_cluster_index = np.zeros(hdb.labels_.shape, dtype=bool)
        for label in cluster_labels:
            index = hdb.labels_ == label

            # (non-noise) clusters with large zspan
            stationary = False
            if label != -1:
                if zspan(index) > min_zspan:
                    stationary = True
                elif zspan(index) > 0.3:
                    if ((starts_near_zmin(index) or ends_near_zmax(index)) and
                        xspan(index) < 1):
                        stationary = True

            a = area(X[index])
            xm = xmean(index)
            ym = ymean(index)

            #if (stationary and
            #    xspan(index) > 1 and
            #    a > area_limit and
            #    not within_bb(xm, ym, OBSTACLE_BB)):
            #    stationary = False

            # If the xy area is large, part of the cyclist trajectory has been
            # grouped into this cluster and we must manually split it.
            # This may require multiple splits and we hardcode a limit of 3
            # iterations.
            split_counter = 0
            while (stationary and
                   a > area_limit and
                   not within_bb(xm, ym, OBSTACLE_BB)):
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

                # increase bounding box by 10% in x/y directions
                dx = xmax - xmin
                dy = ymax - ymin
                xmin -= 0.05*dx
                xmax += 0.05*dx
                ymin -= 0.05*dy
                ymax += 0.05*dy

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

                split_counter += 1
                if split_counter >= 3:
                    break

            cluster_data.append(ClusterData(
                label,
                index,
                (xmean(index), ymean(index), zmean(index)),
                (xspan(index), yspan(index), zspan(index)),
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
                (xmean(index), ymean(index), zmean(index)),
                (xspan(index), yspan(index), zspan(index)),
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
    def _set_radius_mask(X, radius=1.0):
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
            mask = self.bb_mask | ~self.stationary_mask
        else:
            mask = self.bb_mask | self.stationary_mask
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
        size_x = X.count()

        # stationary
        Y = self._compressed_points(lidar_index, True)
        size_y = Y.count()

        # if the difference in sets is greater than 20% of the sum of both sets,
        # probably some errors with clustering
        if abs(size_x - size_y) > 0.2*(size_x + size_y):
            msg = 'Difference in size of bicycle and obstacle cluster '
            msg += 'exceeds 20% of the total set size.\n'
            msg += 'Clustering recomputed with k-means.'
            warnings.warn(msg, UserWarning)

            X, Y = dtc.cluster(*np.concatenate((X, Y)).T)

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

        try:
            assert timestamp >= self.lidar.time[0]
        except AssertionError:
            # Timestamp occurs before the start of lidar data, likely within the
            # first few samples of bicycle data for this event. Lidar frames
            # have a sampling rate of 20 Hz and the largest error we can
            # introduce by shifting the dtc timestamp is 50 ms.
            msg = 'Distance-to-collision timestamp occurs before the '
            msg += 'start of lidar data.\n'
            msg += 'Timestamp: {}\n'.format(timestamp)
            msg += 'Start time of lidar data: {}\n'.format(self.lidar.time[0])
            msg += 'Adjusting dtc timestamp to start time of lidar data.\n'
            warnings.warn(msg, UserWarning)
            return self._calculate_dtc(0)[0]

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

    def _calculate_steer_event_metrics(self):
        if self.type.value != EventType.Overtaking.value:
            raise TypeError('Incorrect EventType')

        region = find_steering_region(self, event_triple=True)
        params = fit_steering_model(self, slice(region.start, region.end))
        return SteerEventGaussianFitParameters(region.start,
                                               region.apex,
                                               region.end,
                                               params[0],
                                               params[1],
                                               params[2],
                                               params[3])

    def _calculate_brake_event_fit(self,
                                   window_size=55,
                                   braking_threshold=0.1,
                                   min_size=75):
        t = self.bicycle['time']
        v = self.bicycle['speed']

        filtered_velocity = ff.moving_average(v, window_size, window_size/2)
        filtered_acceleration = ff.moving_average(
                self.bicycle['accelerometer x'], window_size, window_size/2)

        braking_slice = braking.get_trial_braking_indices(
            filtered_acceleration, braking_threshold, min_size)[0]

        # determine if wheel lockup occurs
        lockup_mask = ((v[braking_slice] < 0.2) &
                       (filtered_acceleration[braking_slice] > 2.5))

        # best-fit line metrics
        slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(
                t[braking_slice][~lockup_mask],
                v[braking_slice][~lockup_mask])

        fitparams = BrakeEventLinearFitParameters(window_size,
                                                  braking_threshold,
                                                  min_size,
                                                  t,
                                                  filtered_velocity,
                                                  filtered_acceleration,
                                                  braking_slice,
                                                  lockup_mask,
                                                  slope,
                                                  intercept,
                                                  r_value,
                                                  p_value,
                                                  stderr)
        return fitparams

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

    def trajectory(self, mode=None, bbmask=None):
        if mode == 'kalman':
            x, y = np.squeeze(
                    self.kalman_smoothed_result.state_estimate[:, :2]).T
            return x, y

        # stationary points
        x = self.x.copy()
        y = self.y.copy()
        x.mask = self.stationary_mask | self.bb_mask
        y.mask = self.stationary_mask | self.bb_mask

        # determine stationary noise bounding boxes
        stationary_bboxes = []
        for cluster in self.clusters:
            if cluster.stationary:
                X = self.valid_points[cluster.index]
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                stationary_bboxes.append({'xlim': (mins[0], maxs[0]),
                                          'ylim': (mins[1], maxs[1])})

        # mask points in specified bounding boxes
        _apply_bbmask(stationary_bboxes, x, y)
        _apply_bbmask(OBSTACLE_BB, x, y)
        if bbmask is not None:
            _apply_bbmask(bbmask, x, y)

        # trajectory points
        xm = x.mean(axis=1)
        ym = y.mean(axis=1)

        # mask elements where the point cloud size is low
        minimum_point_cloud_size = 5
        mask = x.count(axis=1) < minimum_point_cloud_size
        xm[mask] = np.ma.masked
        ym[mask] = np.ma.masked

        if mode is None or mode == 'raw':
            return xm, ym

        # interpolated trajectory
        xm[xm.mask] = np.interp(
                np.where(xm.mask)[0], np.where(~xm.mask)[0], xm[~xm.mask])
        ym[ym.mask] = np.interp(
                np.where(ym.mask)[0], np.where(~ym.mask)[0], ym[~ym.mask])

        # filter out large jumps and re-interp
        mask = np.zeros(xm.shape, dtype=bool)
        # max_xvel is x change in one frame
        # -> (0.4 m)/(0.05 sec) = 28.8 kph
        max_xvel = 0.4
        mask[1:] = np.abs(np.diff(xm)) > max_xvel
        xm[mask] = np.interp(
                np.where(mask)[0], np.where(~mask)[0], xm[~mask])
        ym[mask] = np.interp(
                np.where(mask)[0], np.where(~mask)[0], ym[~mask])

        if mode == 'interp':
            return xm, ym

        # butterworth filtered trajectory
        if mode == 'butter':
            order = 4
            fc = 0.5
            fs = 20
            wn = fc / (0.5*fs)
            b, a = scipy.signal.butter(order, wn, btype='lowpass')
            butterf = lambda x: scipy.signal.filtfilt(b, a, x)
            return butterf(xm), butterf(ym)

        raise ValueError('Unhandled case for mode {}:'.format(mode))

    def plot_trajectory(self, ax=None, plot_vel=True, bbmask=None, **fig_kw):
        if plot_vel:
            num_plots = 2
        else:
            num_plots = 1

        if ax is None:
            fig, ax = plt.subplots(num_plots, 1, sharex=False, **fig_kw)
        else:
            assert len(ax) == num_plots
            fig = ax[0].get_figure()

        if not plot_vel:
            ax = [ax]

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
        if bbmask is None:
            ax[0].scatter(x, y, s=3, marker='.', color=colors[1],
                          label='non-stationary points')
        else:
            mask = _apply_bbmask(bbmask, x, y, apply_mask=False)
            ax[0].scatter(x[~mask], y[~mask],
                          s=3, marker='.',
                          color=colors[1],
                          label='non-stationary points')
            ax[0].scatter(x[mask], y[mask],
                          s=3, marker='.',
                          color=colors[0],
                          label='non-stationary points (masked)')

        # trajectory points
        xm, ym = self.trajectory(mode='raw', bbmask=bbmask)
        ax[0].scatter(xm, ym, s=5, edgecolor=colors[5],
                      label='NSP centroid (per frame)')

        # interpolated trajectory
        xm, ym = self.trajectory(mode='interp', bbmask=bbmask)
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

        ax[0].legend()
        if not plot_vel:
            return fig, ax[0]

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
    def __init__(self, record, bicycle_data, lidar_data, period,
                 event_type, bbmask=None):
            super().__init__(bicycle_data, period)
            self.record = record
            self.bicycle = self.data
            self.lidar = lidar_data

            self.event_detection = None
            self.event = None
            self._detect_event(event_type, bbmask=bbmask)

    @staticmethod
    def detect_valid_index(lidar_data, bbmask=None, count=None):
        x, y, z = lidar_data.cartesianz(**VALID_BB)

        _apply_bbmask(OBSTACLE_BB, x, y, z)
        if bbmask is not None:
            _apply_bbmask(bbmask, x, y, z)

        if count is None:
            count = 1
        return x.count(axis=1) > count

    def _detect_event(self, event_type, bbmask=None):
        """The event is detected using the following masks: VALID_BB, ENTRY_BB,
        EXIT_BB_BRAKE, and EXIT_BB_STEER. Candidates for the event are first
        determined detecting the cyclist withing VALID_BB. The candidates are
        searched for ENTRY_BB and EXIT_BB_BRAKE, EXIT_BB_STEER to determine the
        most likely event. Use of EXIT_BB_STEER or EXIT_BB_BRAKE is determined
        by the event_type argument.

        Parameters:
        bbmask: dict or iteratble of dicts, keywords supplied to
                lidar.cartesian() for an area to ignore for event detection.
                This is used in the event of erroneous lidar data.
        """
        # determine event indices from lidar data
        # increase the count threshold if all indices are valid
        count = 1
        while True:
            valid_index = Trial2.detect_valid_index(self.lidar, bbmask, count)

            # require at least 20% of indices to be non-valid
            if np.sum(valid_index)/valid_index.size < 0.8:
                break
            count += 1
        event_clumps = np.ma.extras._ezclump(valid_index)

        # if clumps are separated by 3 or fewer indices, combine them
        for i in reversed(range(1, len(event_clumps))):
            a = event_clumps[i - 1]
            b = event_clumps[i]
            if b.start - a.stop < 3:
                event_clumps[i - 1] = slice(a.start, b.stop)
                event_clumps.pop(i)

        # filter out events with a minimum size
        MIN_TIME_DURATION = int(5.5/np.diff(self.lidar.time).mean()) # in samples
        event_clumps = [c for c in event_clumps
                        if c.stop - c.start > MIN_TIME_DURATION]
        # filter out events that end in the first third of the trial
        event_clumps = [c for c in event_clumps
                        if c.stop > 0.3*valid_index.size]

        msg = "unable to detect event for this trial (count = {})".format(count)
        assert len(event_clumps) > 0, msg

        # reduce span of event using entry and exit bounding box detection
        def bbox_clumps(bbox, slice_):
            index = self.lidar.cartesian(**bbox)[0].count(axis=1) > 1

            mask = np.zeros(index.shape, dtype=bool)
            mask[slice_.start:slice_.stop] = True

            index &= mask
            return np.ma.extras._ezclump(index)

        def first_clump_in_slice(clumps, slice_, clump_edge=None):
            """Check if clump is contained withing slice. If clump_edge is not
            None, check only if the specified edge is contained.
            """
            for c in clumps:
                if clump_edge is None:
                    if c.start > slice_.start and c.stop < slice_.stop:
                        return c
                else:
                    e = getattr(c, clump_edge)
                    if e > slice_.start and e < slice_.stop:
                        return c
            return None

        # use last clump for the event
        # search backwards in event clumps for entry/exit conditions
        for event_index in event_clumps[::-1]:

            # find time where cyclist enters lidar vision
            # use falling edge of first entry clump within the event slice
            entry_clumps = bbox_clumps(ENTRY_BB, event_index)
            c = first_clump_in_slice(entry_clumps, event_index, 'stop')
            entry_time = None
            if c is not None:
                event_index = slice(c.stop, event_index.stop)
                entry_time = self.lidar.time[c.stop]

            # find time where cyclist has finished overtaking obstacle, if it
            # exists using the rising edge of the first clump within the event
            # slice
            if event_type == EventType.Overtaking:
                exit_clumps = bbox_clumps(EXIT_BB_STEER, event_index)
            elif event_type == EventType.Braking:
                exit_clumps = bbox_clumps(EXIT_BB_BRAKE, event_index)
            else:
                raise ValueError(event_type)

            c = first_clump_in_slice(exit_clumps, event_index, 'start')
            exit_time = None
            if c is not None:
                event_index = slice(event_index.start, c.start)
                exit_time = self.lidar.time[c.start]

            if entry_time is not None and exit_time is not None:
                # event found, otherwise check next event clump
                break

        if entry_time is None:
            entry_time = self.lidar.time[event_index.start]
            msg = 'Unable to detect cyclist entry for event starting at '
            msg += 't = {0:.3f} seconds'.format(
                    self.lidar.time[event_index.start])
            warnings.warn(msg, UserWarning)

        if exit_time is None:
            exit_time = self.lidar.time[event_index.stop - 1]
            msg = 'Unable to detect cyclist exit or braking for event '
            msg += 'ending at t = '
            msg += '{0:.3f} seconds'.format(
                    self.lidar.time[event_index.stop - 1])
            warnings.warn(msg, UserWarning)

        if event_type == EventType.Overtaking:
            exit_brake_clumps = []
            exit_steer_clumps = exit_clumps
        else:
            exit_brake_clumps = exit_clumps
            exit_steer_clumps = []

        self.event_detection = EventDetectionData(
            valid_clumps=event_clumps,
            entry_clumps=entry_clumps,
            exit_brake_clumps=exit_brake_clumps,
            exit_steer_clumps=exit_steer_clumps,
            event_slice=event_index,
            entry_time=entry_time,
            exit_time=exit_time)

        # get indices for bicycle time
        i0 = np.where(self.bicycle.time > entry_time)[0][0]
        if i0 != 0:
            i0 -= 1

        try:
            i1 = np.where(self.bicycle.time > exit_time)[0][0]
        except IndexError:
            i1 = -1

        self.event = Event(
                self,
                self.bicycle[i0:i1],
                self.lidar[event_index],
                self.period,
                event_type,
                bbmask=bbmask)

def _apply_bbmask(bounding_box, x, y, z=None, apply_mask=True):
    mask = np.zeros(x.shape, dtype=bool)

    if isinstance(bounding_box, dict):
        bounding_box = [bounding_box]

    for bb in bounding_box:
        bbmask = np.ones(x.shape, dtype=bool)
        if 'xlim' in bb:
            bbmask &= ((x < bb['xlim'][1]) &
                       (x > bb['xlim'][0]))
        if 'ylim' in bb:
            bbmask &= ((y < bb['ylim'][1]) &
                       (y > bb['ylim'][0]))
        if 'zlim' in bb:
            bbmask &= ((z < bb['zlim'][1]) &
                       (z > bb['zlim'][0]))
        mask |= bbmask

    if apply_mask:
        x[mask] = np.ma.masked
        y[mask] = np.ma.masked
        if z is not None:
            z[mask] = np.ma.masked
    return mask


def find_steering_region(event, event_triple=False, ax=None, obstacle_origin=False):
    EventTriple = namedtuple('EventTriple',
                             ['start', 'apex', 'end'])
    class Extremum(enum.Enum):
        MINIMUM = -1
        INFLECT = 0
        MAXIMUM = 1

        def __str__(self):
            return self.name

    class ExtremumPoint(object):
        def __init__(self, event_index, extremum, x, y):
            self.index = event_index
            self.extremum = extremum
            self.x = x
            self.y = y

        def __eq__(self, other):
            # order by x-coordinate
            # This assumes two different extrema cannot have the same
            # x-coordinate.
            return self.x == other.x

        def __lt__(self, other):
            # order by x-coordinate
            return self.x < other.x

        def __str__(self):
            msg = '[{}, {}, ({:0.3f}, {:0.3f})]'.format(
                    self.index, self.extremum, self.x, self.y)
            return msg

        __repr__ = __str__

    # get kalman estimate trajectory
    x, y = event.trajectory(mode='kalman')

    # smooth y-coordinate and get local extrema
    fy = scipy.signal.savgol_filter(y, 51, 3)
    maxima_index = scipy.signal.argrelmax(fy)[0]
    minima_index = scipy.signal.argrelmin(fy)[0]

    # smooth dy and get inflection points
    dy = scipy.signal.savgol_filter(np.diff(y), 85, 3)
    inflec_index = np.concatenate((
        scipy.signal.argrelmax(dy)[0],
        scipy.signal.argrelmin(dy)[0]))

    if maxima_index.size == 0:
        # try finding maxima using the double derivative of y
        ddy = scipy.signal.savgol_filter(np.diff(dy), 125, 3)

        # require height limit as the double derivative is more sensitive
        maxima_index = scipy.signal.argrelmin(ddy)[0]
        ddy_ptp = ddy.ptp()
        maxima_index = np.array([elem for elem in maxima_index
                                 if ddy[elem] < -ddy_ptp/10])

        if ax is not None:
            colors = sns.color_palette('tab10', 10)
            ax.plot(x[:-2], ddy*y.ptp()/ddy_ptp + y.mean(),
                    label='ddy (scaled & shifted)',
                    color=colors[7])
            ax.axhline(y.mean())

    # Create lists of ExtremumPoints for minima, maxima, inflection points and
    # then combine into a single sorted list. The y-coordinate is the Kalman
    # estimate trajectory instead of the smoothed one used for extremum
    # detection.
    maxima = [ExtremumPoint(i, Extremum.MAXIMUM, x[i], y[i])
              for i in maxima_index]
    minima = [ExtremumPoint(i, Extremum.MINIMUM, x[i], y[i])
              for i in minima_index]
    inflec = [ExtremumPoint(i, Extremum.INFLECT, x[i], y[i])
              for i in inflec_index]
    points = maxima + minima + inflec
    points.sort()
    #print(points)

    # Find start by searching for the first maximum to the right (+x direction)
    # of the obstacle centroid.
    obstacle = ExtremumPoint(None,
                             None,
                             OBSTACLE_POINT[0],
                             None)
    i = bisect.bisect_right(points, obstacle)
    #print(i)

    def argfind(points, extremum_type, start_index=None, stop_index=None,
                reverse=False):
        if start_index is None:
            offset = 0
        else:
            offset = start_index

        iterator = enumerate(points[start_index:stop_index])
        if reverse:
            iterator = reversed(list(iterator))
        for i, p in iterator:
            if p.extremum == extremum_type:
                return i + offset
        raise ValueError

    arg_max = argfind(points, Extremum.MAXIMUM, start_index=i)
    #print('first maximum, event start')
    #print(arg_max, points[arg_max])

    arg_min = argfind(points, Extremum.MINIMUM, stop_index=arg_max,
                      reverse=True)
    #print('first minimum, event apex')
    #print(arg_min, points[arg_min])

    #print('corresponding trajectory point, event stop')
    #print('arg min index', points[arg_min].index)
    #print('total size', len(x))
    # as index in y array increases, cyclist moves to the left (-x direction)
    index_end = (points[arg_min].index - 1 +
                 np.searchsorted(y[points[arg_min].index:],
                                 points[arg_max].y))

    #print('({:0.3f}, {:0.3f})'.format(x[index_end], y[index_end]))

    index_start = points[arg_max].index

    if ax is not None:
        colors = sns.color_palette('tab10', 10)

        if obstacle_origin:
            x0 = -OBSTACLE_POINT[0]
            y0 = -OBSTACLE_POINT[1]
        else:
            x0 = 0
            y0 = 0

        # plot trajectory, dy
        ax.plot(x + x0,
                y + y0,
                color=colors[0],
                label='Kalman estimate trajectory')
        #ax.plot(x[:-1] + x0,
        #        dy*y.ptp()/dy.ptp() + y.mean() + y0,
        #        color=colors[2], alpha=0.8,
        #        label='dy (scaled & shifted)')

        # plot extrema
        extrema_marker_size = 80
        ax.scatter(x[maxima_index] + x0,
                   y[maxima_index] + y0,
                   label='maxima',
                   s=extrema_marker_size,
                   marker='^',
                   color=colors[1])
        ax.scatter(x[minima_index] + x0,
                   y[minima_index] + y0,
                   label='minima',
                   s=extrema_marker_size,
                   marker='v',
                   color=colors[1])
        #ax.scatter(x[inflec_index] + x0,
        #           y[inflec_index] + y0,
        #           label='inflection points',
        #           s=extrema_marker_size,
        #           marker='d',
        #           color=colors[1])

        # plot event points
        event_point_marker_size = 150
        event_point_linewidth = 3
        ax.scatter(points[arg_max].x + x0,
                   points[arg_max].y + y0,
                   label='event start',
                   s=event_point_marker_size,
                   marker='^',
                   linewidth=event_point_linewidth,
                   facecolor='None',
                   edgecolor=colors[3])
        #ax.scatter(points[arg_min].x + x0,
        #           points[arg_min].y + y0,
        #           label='event apex',
        #           s=event_point_marker_size,
        #           marker='v',
        #           linewidth=event_point_linewidth,
        #           facecolor='None',
        #           edgecolor=colors[3])
        ax.scatter(x[index_end] + x0,
                   y[index_end] + y0,
                   label='event end',
                   s=event_point_marker_size,
                   marker='s',
                   linewidth=event_point_linewidth,
                   facecolor='None',
                   edgecolor=colors[3])

        # plot event regions
        ax.axvspan(x[index_end] + x0,
                   x[index_start] + x0,
                   label='overtaking event region (trajectory)',
                   hatch='X', fill=False,
                   color=colors[3], alpha=0.3)

        ax.legend()

    if event_triple:
        return EventTriple(index_start, points[arg_min].index,index_end)
    return slice(index_start, index_end)


def fit_steering_model(event,
                       event_slice,
                       ax=None,
                       obstacle_origin=False,
                       **plot_kw):
    x, y = event.trajectory(mode='kalman')
    x_ = x[event_slice]
    y_ = y[event_slice]

    def gauss(x, *params):
        amplitude, mu, sigma, offset = params
        return amplitude*np.exp(-(x-mu)**2/(2*sigma**2)) + offset

    initial_guess = [-y_.ptp(),
                     x_[np.argmin(y_)],
                     x_.ptp()/3,
                     OBSTACLE_POINT[1]]
    params, _ = scipy.optimize.curve_fit(gauss, x_, y_, p0=initial_guess)

    if ax is not None:
        if obstacle_origin:
            x0 = -OBSTACLE_POINT[0]
            y0 = -OBSTACLE_POINT[1]
        else:
            x0 = 0
            y0 = 0

        colors = sns.color_palette('tab10', 10)
        if plot_kw:
            ax.plot(x_ + x0,
                    gauss(x_, *params) + y0,
                    **plot_kw)
        else:
            ax.plot(x_ + x0,
                    gauss(x_, *params) + y0,
                    label='gaussian fit',
                    color=colors[0],
                    linestyle='--')
        ax.legend()
    return params
