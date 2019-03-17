# -*- coding: utf-8 -*-
import queue
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def dist(a, b=None):
    if b is None:
        # assume 'a' is a pair
        assert len(a) == 2
        b = a[1]
        a = a[0]

    ax, ay = a
    bx, by = b
    return np.sqrt((ax - bx)**2 + (ay - by)**2)


def bcp(cluster_a, cluster_b, lateral_calc=False):
    min_dist = np.inf
    pair = None

    # calculate centroid-to-centroid distance
    a_centroid = cluster_a.mean(axis=0)
    b_centroid = cluster_b.mean(axis=0)
    dist_centroid = dist(a_centroid, b_centroid)

    for a in cluster_a:
        for b in cluster_b:
            d = dist(a, b)

            # ignore some points if we are calculating lateral clearance
            if lateral_calc:
                # ignore if y dist is too small
                if abs(b[1] - a[1]) <  0.3:
                    continue

                # ignore if point is too close to the opposite cluster centroid
                if dist(a, b_centroid) < 0.5:
                    continue

                # ignore if point is too close to the opposite cluster centroid
                if dist(b, a_centroid) < 0.5:
                    continue

            if d < min_dist:
                min_dist = d
                pair = (a, b)

    return pair

def cluster(x, y, initial_guess=None):
    try:
        x = x.compressed()
        y = y.compressed()
    except AttributeError:
        pass

    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    X = np.vstack((x, y)).transpose()

    if initial_guess is None:
        init = 'k-means++'
    else:
        init = initial_guess
    kmeans = KMeans(n_clusters=2, init=init).fit(X)

    index0 = kmeans.labels_ == 0
    index1 = kmeans.labels_ == 1

    # set indices such that cluster A is on the left
    if x[index0][0] >= x[index1][0]:
        index0 = kmeans.labels_ == 1
        index1 = kmeans.labels_ == 0

    cluster_a = np.ma.array(list(zip(x[index0], y[index0])))
    cluster_b = np.ma.array(list(zip(x[index1], y[index1])))
    return cluster_a, cluster_b

def plot_closest_pair(cluster_a, cluster_b,
                      pair=None, colors=None, ax=None, **kwargs):
    if colors is None:
        colors = sns.color_palette('Paired', 10)[1::2]

    return_pair = False
    if pair is None:
        # need to calculate pair
        pair = bcp(cluster_a, cluster_b)
        return_pair = True

    if ax is None:
        _, ax = plt.subplots(**kwargs)

    # plot clusters
    ax.plot(*cluster_a.T,
            linestyle='None',
            marker='.',
            color=colors[0],
            label='cluster A')
    ax.plot(*cluster_b.T, marker='.',
            linestyle='None',
            color=colors[1],
            label='cluster B')

    # plot closest pair with different markers
    ax.plot(*pair[0],
            linestyle='None',
            marker='o', markersize=10,
            markerfacecolor='None',
            color=colors[2],
            label='closest pair A')
    ax.plot(*pair[1],
            linestyle='None',
            marker='o', markersize=10,
            markerfacecolor='None',
            color=colors[2],
            label='closest pair B')

    # plot line connecting closest pair
    ax.plot(*np.vstack(pair).T,
            color=colors[2],
            label='closest pair line')

    ax.legend()

    if return_pair:
        return ax, pair
    return ax
