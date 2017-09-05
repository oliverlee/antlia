#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def kalman(Ad, Bd, Cd, Q, R, z, u=None, x0=None, P0=None,
           missed_measurement=[]):
    n = z.shape[0]

    xhat = np.zeros((n, Ad.shape[0], 1))
    xhatminus = np.zeros(xhat.shape)
    P = np.zeros((n,) + Ad.shape)
    Pminus = np.zeros(P.shape)
    K = np.zeros((n,) + tuple(reversed(Cd.shape)))

    if x0 is None:
        x0 = np.zeros((Ad.shape[0], 1))
        P0 = np.zeros(Ad.shape)

    for i in range(n):
        # time update
        if i == 0:
            xhatminus[i, :] = np.dot(Ad, x0)
            Pminus[i, :] = np.dot(np.dot(Ad, P0), Ad.T) + Q
        else:
            # time update state
            xhatminus[i, :] = np.dot(Ad, xhat[i - 1, :])
            # time update error covariance
            Pminus[i, :] = np.dot(np.dot(Ad, P[i - 1, :]), Ad.T) + Q

        # measurement update
        # measurement update kalman gain
        S = np.dot(np.dot(Cd, Pminus[i, :]), Cd.T) + R
        if i in missed_measurement:
            S += 99*R
        K[i, :] = np.linalg.lstsq(S, np.dot(Cd, Pminus[i, :].T))[0].T
        # measurement update state
        xhat[i, :] = (xhatminus[i, :] +
                      np.dot(K[i, :], (z[i, :] - np.dot(Cd, xhatminus[i, :]))))
        P[i, :] = np.dot(np.eye(Ad.shape[0]) - np.dot(K[i, :], Cd), Pminus[i, :])
    return xhat, P, K

def kalman_velocity(dt, v, u, z, q=5, missed_measurement=[]):
    Ad = np.array([
        [1, dt],
        [0,  1]
    ])
    Bd = np.array([
        [1/2*dt**2],
        [dt],
    ])
    Cd = np.array([
        [0, 1],
    ])
    Q = q * np.array([
        [1/4*dt**4, 1/2*dt**3],
        [1/2*dt**3,    dt**2]
    ])
    """
    Find variance at 'constant' speed section
    In [12]: r['speed'][np.argwhere(
        r['time'] > 16)[0][0]:np.argwhere(r['time'] > 21)[0][0]].var()
    Out[12]: 0.64835952938689101
    """
    R = np.array([
        [0.6483595]
    ])
    u = np.reshape(u, (-1, 1, 1))
    z = np.reshape(z, (-1, 1, 1))

    xhat, P, K = kalman(Ad, Bd, Cd, Q, R, z, u,
                        missed_measurement=missed_measurement)
    return np.squeeze(xhat[:, 1])
