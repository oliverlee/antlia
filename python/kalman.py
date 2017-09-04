#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def kalman(Ad, Bd, Cd, Q, R, z, u=None, x0=None, P0=None):
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
        K[i, :] = np.linalg.lstsq(S, np.dot(Cd, Pminus[i, :].T))[0].T
        # measurement update state
        xhat[i, :] = (xhatminus[i, :] +
                      np.dot(K[i, :], (z[i, :] - np.dot(Cd, xhatminus[i, :]))))
        P[i, :] = np.dot(np.eye(Ad.shape[0]) - np.dot(K[i, :], Cd), Pminus[i, :])
    return xhat, P, K
