# -*- coding: utf-8 -*-
from enum import Enum
from collections import namedtuple
import heapq
import itertools
import warnings

import numpy as np
import scipy.signal
import scipy.spatial
import scipy.stats
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
        ['label', 'index', 'zmean', 'zspan', 'count', 'area', 'stationary'])

FakeHdb = namedtuple(
        'FakeHdb',
        ['labels_'])

SteeringIdentificationCase = namedtuple(
        'SteeringIdentificationCase',
        ['attenuation', 'data', 'freq', 'xf', 'minima', 'maxima', 'inflections',
         'cutoff', 'section', 'score'])

SteerEventSinusoidFitParameters = namedtuple(
        'SteerEventSinusoidFitParameters',
        ['amplitude',
         'frequency',
         'phase',
         'mean']) # TODO add all information from
                  # steering identification case object?

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


class EventType(Enum):
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

            self.si = None
            self.steer_slice = None # bicycle time
            #self._identify_steer_slice() # don't run automatically

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
        if len(cluster_labels) > 100:
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

        extra_cluster_index = np.zeros(hdb.labels_.shape, dtype=bool)
        for label in cluster_labels:
            index = hdb.labels_ == label

            # (non-noise) clusters with large zspan
            stationary = (label != -1 and
                          (zspan(index) > min_zspan*z.shape[0] or
                           (zspan(index) > 0.3*z.shape[0] and
                            area(X[index]) < area_limit)))

            # if the xy area is large, part of the cyclist trajectory has been
            # grouped into this cluster and we must manually split it
            if stationary and area(X[index]) > area_limit:
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

    def _identify_steer_slice(self, pattern=None, attenuation=None,
                              min_allowed_freq=0.40):
        if self.type != EventType.Overtaking:
            raise TypeError('Incorrect EventType')

        if attenuation is None:
            attenuation = np.arange(35, 100, 10)

        # create windows of pattern length
        if pattern is None:
            test_patterns = [(0, -1, 0, 1, 0, -1, 0),
                             (0, -1, 0, 1, 0)]
        else:
            test_patterns = [pattern]

        # save identification data
        self.si = SteeringIdentification(attenuation.copy(),
                                         list(test_patterns))

        best_a = None
        for a in attenuation:
            freq, xf = self._chebwin_fft(self.bicycle['steer angle'],
                                         self.period,
                                         attenuation=a,
                                         max_freq=2)
            minima = scipy.signal.argrelmin(xf)[0]
            if len(minima) < 2:
                # Unable to detect minima for this attenuation value
                continue

            f = freq[minima[[0, 1]]]

            if f[0] > min_allowed_freq:
                msg = 'First frequency minimum exceeds {}'.format(
                        min_allowed_freq)
                msg += ' and has been replaced with 0.1'
                warnings.warn(msg)
                f[1] = f[0]
                f[0] = 0.1

            sa = self._butter_bandpass_filter(self.data['steer angle'],
                                              f.copy(),
                                              1/self.period)

            # get max, min, inflection points for bandpass filtered steer angle
            sa_max = scipy.signal.argrelmax(sa)[0]
            sa_min = scipy.signal.argrelmin(sa)[0]
            dsa = np.diff(sa)
            sa_inf = np.concatenate((scipy.signal.argrelmax(dsa)[0],
                                     scipy.signal.argrelmin(dsa)[0]))

            # put max, min, inf in index order
            q = []
            for item in sa_max:
                heapq.heappush(q, (item, 1))
            for item in sa_min:
                heapq.heappush(q, (item, -1))
            for item in sa_inf:
                heapq.heappush(q, (item, 0))

            sorted_q = []
            while q:
                sorted_q.append(heapq.heappop(q))

            # reduce runs of 3 to 1 (only happens with inflection points)
            q = []
            for item in sorted_q:
                q.append(item)

                if len(q) > 2:
                    if q[-1][1] == q[-2][1] == q[-3][1]:
                        q.pop(-3)
                        q.pop(-1)

            sections = []
            # filter sections that match pattern
            for p in test_patterns:
                # create window iterator
                it = itertools.tee(q, len(p))
                for n, l in enumerate(it):
                    for _ in range(n):
                        next(l, None)
                window = zip(*it)

                # filter pattern
                for w in window:
                    w0, w1 = list(zip(*w))
                    if w1 == p:
                        sections.append(w0)

            # filter sections that span the width of the obstacle
            def span_obs(s):
                t0, t1 = self.bicycle.time[[s[0], s[-1]]]
                z = self.lidar.cartesianz()[2]
                z[(z < t0) | (z > t1)] = np.ma.masked
                z.mask |= self.bb_mask | self.stationary_mask
                x = np.ma.masked_where(z.mask, self.x, copy=True)
                xlim = OBSTACLE_BB['xlim']
                return x.max() > xlim[1] and x.min() < xlim[0]
            sections = filter(span_obs, sections)

            # define best section to have the longest time duration
            best_section = None
            for s in sections:
                score = s[-1] - s[0]
                if best_section is None or score > best_section[1]:
                    best_section = (s, score)

            if best_section is None:
                # didn't find acceptable section for bandpass attenuation value
                self.si.add_case(a, sa, freq, xf, sa_min, sa_max, sa_inf, f,
                                 None, None)
                continue
            else:
                self.si.add_case(a, sa, freq, xf, sa_min, sa_max, sa_inf, f,
                                 best_section[0], best_section[1])

            if best_a is None or best_section[1] > best_a[1]:
                best_a = (best_section[0], best_section[1], a)

        msg = 'Unable to determine steering metrics for event'
        assert best_a is not None, msg
        self.steer_slice = slice(best_a[0][0], best_a[0][-1] + 1)

    def _calculate_steer_event_metrics(self):
        if self.type != EventType.Overtaking:
            raise TypeError('Incorrect EventType')
        if self.si is None:
            raise ValueError('Steer slice must first be identified')

        case = None
        for c in self.si.cases:
            if case is None or c.score > case.score:
                case = c

        duration = (self.bicycle.time[case.section[-1]] -
                self.bicycle.time[case.section[0]])
        amplitude = (case.data[case.section[3]] -
                case.data[case.section[1]])
        length = len(case.section)
        return duration, amplitude, length

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

    def plot_steer_identification(self, steerid_kw=None, ax=None, **fig_kw):
        if self.type != EventType.Overtaking:
            raise TypeError('Incorrect EventType')
        if self.si is None:
            if steerid_kw is None:
                steerid_kw = {}
            self._identify_steer_slice(**steerid_kw)

        if ax is None:
            fig, ax = plt.subplots(4, 1, sharex=False, **fig_kw)
        else:
            assert len(ax) == 4
            fig = ax[0].get_figure()

        n = len(self.si.cases)
        if n < 11:
            colors = sns.color_palette('tab10', n)
        else:
            colors = sns.husl_palette(n, l=.7)

        lines = []
        case_1_shift = 0.025
        case_2_shift = 0.25
        for i, case in enumerate(self.si.cases):
            # steer angle fft plot
            label = 'attenuation = {}'.format(case.attenuation)
            ax[0].plot(case.freq, case.xf, color=colors[i], label=label)

            minima = scipy.signal.argrelmin(case.xf)[0]
            ax[0].scatter(case.freq[minima], case.xf[minima],
                          marker='v', color=colors[i])

            # filtered steer angle plot
            label += ', fc = [{:0.3f}, {:0.3f}] Hz'.format(*case.freq)
            if case.score is not None:
                label += ', span = {}'.format(case.score)
            lines.extend(ax[1].plot(self.bicycle.time,
                                    case.data + case_1_shift*(i + 1),
                                    label=label, color=colors[i]))
            ax[1].scatter(self.bicycle.time[case.maxima],
                          case.data[case.maxima] + case_1_shift*(i + 1),
                          marker='^', color=colors[i])
            ax[1].scatter(self.bicycle.time[case.minima],
                          case.data[case.minima] + case_1_shift*(i + 1),
                          marker='v', color=colors[i])
            ax[1].scatter(self.bicycle.time[case.inflections],
                          case.data[case.inflections] + case_1_shift*(i + 1),
                          marker='d', color=colors[i])

            try:
                t0, t1 = self.bicycle.time[[case.section[0], case.section[-1]]]
            except IndexError:
                pass
            else:
                ax[1].axvspan(t0, t1, alpha=0.1, color=colors[i])

                # trajectory scatter plot
                z = self.lidar.cartesianz()[2]
                z[(z < t0) | (z > t1)] = np.ma.masked
                z.mask |= self.bb_mask | self.stationary_mask
                x = np.ma.masked_where(z.mask, self.x, copy=True)
                y = np.ma.masked_where(z.mask, self.y, copy=True)
                ax[2].scatter(x, y + case_2_shift*(i + 1),
                              marker='.', color=colors[i], alpha=0.1)

        best_score = None
        for i, case in enumerate(self.si.cases):
            if best_score is None or case.score > best_score[0]:
                best_score = (case.score, i)
        lines[best_score[1]].set_linewidth(3)

        ax[0].set_title('steer angle FFT chebwin window')
        ax[0].set_xlabel('frequency [Hz]')
        ax[0].set_ylabel('amplitude')
        ax[0].legend(loc='upper right')
        ax[0].set_ylim((-0.001, 0.02))

        sa_raw = self.bicycle['steer angle'].copy()
        sa_raw -= sa_raw.mean()
        ax[1].plot(self.bicycle.time, sa_raw,
                   label='measured steer angle (mean subtracted)',
                   color='black', alpha=0.5, zorder=-1)
        ax[1].set_title('filtered steer angle')
        ax[1].set_xlabel('time [s]')
        ax[1].set_ylabel('steer angle (case shifted) [rad]')
        ax[1].legend(loc='upper left')

        ax[2].scatter(self.x, self.y,
                      marker='.', color='black', alpha=0.11,
                      zorder=-1)
        ax[2].set_title('lidar scans (section bounds)')
        ax[2].set_xlabel('x-coordinate [m]')
        ax[2].set_ylabel('y-coordinate [m]')

        # add 3d cluster scatter plot
        ax[3].get_xaxis().set_visible(False)
        ax[3].get_yaxis().set_visible(False)
        ax[3] = fig.add_subplot(4, 1, 4, projection='3d')
        self.plot_clusters(ax=ax[3])

        z0 = self.z.min()
        z1 = self.z.max()
        t0 = self.lidar.time[0]
        t1 = self.lidar.time[-1]

        tstart = self.bicycle.time[self.si.cases[best_score[1]].section[0]]
        tstop = self.bicycle.time[self.si.cases[best_score[1]].section[-1]]
        zstart, zstop = np.interp([tstart, tstop], [t0, t1], [z0, z1])
        index = (self.z < zstart) | (self.z > zstop)
        x = np.ma.masked_where(index, self.x, copy=True)
        y = np.ma.masked_where(index, self.y, copy=True)
        z = np.ma.masked_where(index, self.z, copy=True)
        ax[3].scatter(x, y, z, s=10, color=colors[best_score[1]], alpha=0.5)

        return fig, ax


class Trial2(Trial):
    def __init__(self, record, bicycle_data, lidar_data, period,
                 bbmask=None):
            super().__init__(bicycle_data, period)
            self.record = record
            self.bicycle = self.data
            self.lidar = lidar_data

            self.event_detection = None
            self.event = None
            self._detect_event(bbmask=bbmask)

    @staticmethod
    def detect_valid_index(lidar_data, bbmask=None, count=None):
        x, y, z = lidar_data.cartesianz(**VALID_BB)

        _apply_bbmask(OBSTACLE_BB, x, y, z)
        if bbmask is not None:
            _apply_bbmask(bbmask, x, y, z)

        if count is None:
            count = 1
        return x.count(axis=1) > count

    def _detect_event(self, bbmask=None):
        """The event is detected using the following masks: VALID_BB, ENTRY_BB,
        EXIT_BB_BRAKE, and EXIT_BB_STEER. Candidates for the event are first
        determined detecting the cyclist withing VALID_BB. The candidates are
        searched for ENTRY_BB and EXIT_BB_BRAKE, EXIT_BB_STEER to determine the
        most likely event. Detection of the cyclist in EXIT_BB_STEER is higher
        precedence than EXIT_BB_BRAKE and determines the event type.

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

        # filter out events with a minimum size
        MIN_TIME_DURATION = int(5.5/np.diff(self.lidar.time).mean()) # in samples
        event_clumps = [c for c in event_clumps
                        if c.stop - c.start > MIN_TIME_DURATION]

        assert len(event_clumps) > 0, "unable to detect event for this trial"

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

            # find time where cyclist has finished overtaking obstacle, if it exists
            # using the rising edge of the first clump within the event slice
            # steering exit conditions are prioritized over braking exit conditions
            exit_steer_clumps = bbox_clumps(EXIT_BB_STEER, event_index)

            if exit_steer_clumps:
                event_type = EventType.Overtaking
                exit_brake_clumps = []
            else:
                event_type = EventType.Braking
                exit_brake_clumps = bbox_clumps(EXIT_BB_BRAKE, event_index)

            c = first_clump_in_slice(exit_steer_clumps, event_index, 'start')
            exit_time = None
            if c is not None:
                event_index = slice(event_index.start, c.start)
                exit_time = self.lidar.time[c.start]
            else:
                c = first_clump_in_slice(exit_brake_clumps, event_index, 'start')
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
        if 'zlim' in bounding_box:
            bbmask &= ((z < bb['zlim'][1]) &
                       (z > bb['zlim'][0]))
        mask |= bbmask

    if apply_mask:
        x[mask] = np.ma.masked
        y[mask] = np.ma.masked
        if z is not None:
            z[mask] = np.ma.masked
    return mask
