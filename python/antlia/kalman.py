#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import sys

import numpy as np
import sympy
import sympy.physics.vector as vec
import matplotlib.pyplot as plt
import scipy.integrate
import seaborn as sns

from antlia import filter as ff


def generate_measurement(event):
    z = np.ma.zeros((event.bicycle.time.shape[0], 4))
    z[:, :2] = np.ma.masked

    gyroz = -event.bicycle['gyroscope z']
    accx = -event.bicycle['accelerometer x']
    z[:, 2] = gyroz
    z[:, 3] = accx

    # get x, y quick trajectory estimate from event using
    # butterworth low-pass filter to remove pedaling effects
    x, y = event.trajectory('butter')

    # since lidar/position is sampled at a lower rate than the imu/bicycle
    # sensors, get the corresponding bicycle timestamps for the lidar scans
    index = np.searchsorted(event.bicycle.time, event.lidar.time) - 1
    z[index, 0] = x
    z[index, 1] = y
    return z


def generate_fhFH(constant_velocity=True,
                  sample_period=1/125,      # seconds
                  wheelbase=1.0,            # meters
                  yaw_rate_tolerance=0.01): # radians/sec,
                                            # - switching between f/F cases
    """Generate functions f, h and respective Jacobians F, H for use with an
    Extended Kalman Filter. For now, noise functions W, V are ignored and the
    EKF formulation uses constant noise covariance matrices.

    Parameters:
    constant_velocity: Use a constant velocity model if True or a constant
                       acceleration model if False. Defaults to True.
    sample_period: Duration for each time step.
    wheelbase: Bicycle wheelbase. The system defines point (px, py) to be the
               rear wheel contact point but the lidar-derived trajectory uses
               the (roughly estimated) cyclist centroid. Half the wheelbase is
               considered to be the offset between the two.
    yaw_rate_tolerance: The tolerance used a zero-value yaw rate.

    Sources:
    http://campar.in.tum.de/Chair/KalmanFilter
    http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf
    https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb?create=1
    https://winfriedauner.de/projects/unscented/ctrv/
    """
    px, py, yaw, v = vec.dynamicsymbols('p_x, p_y, psi, v')

    # dv_k, dyaw_k are defined with nonzero=True to prevent integration failure
    # due to some bug
    v_k = sympy.Symbol('v_k', real=True)
    yaw_k = sympy.Symbol('\psi_k', real=True)
    dv_k = sympy.Symbol('\dot{v}_k', real=True, nonzero=True)
    dyaw_k = sympy.Symbol('\dot{\psi}_k', real=True, nonzero=True)

    t_k, t_k1 = sympy.symbols('t_k, t_k+1', real=True)
    t = vec.dynamicsymbols._t
    dt = sympy.Symbol('\Delta t', real=True)

    x = [px, py, yaw_k, v_k, dyaw_k, dv_k]
    dx_k = sympy.Matrix([
        v*sympy.cos(yaw),
        v*sympy.sin(yaw),
        dyaw_k,
        dv_k,
        0,
        0
    ])

    # term integrand for a single time step
    term_integrands = {
        yaw: yaw_k + dyaw_k*(t - t_k),
    }
    if constant_velocity:
        term_integrands[v] = v_k
    else:
        term_integrands[v] = v_k + dv_k*(t - t_k)

    # define state transition function f
    # two cases for f, dyaw_k = 0 and dyaw_k != 0
    df_0 = sympy.integrate(
            dx_k.subs(term_integrands).subs(dyaw_k, 0),
            (t, t_k, t_k1))
    df_1 = sympy.integrate(
            dx_k.subs(term_integrands),
            (t, t_k, t_k1))

    # generate Jacobian F for both cases
    dF_0 = df_0.jacobian(x)
    dF_1 = df_1.jacobian(x)

    # substitute sample period, dt
    simplify_term_dt = lambda x: x.subs({t_k1: dt + t_k}).expand()
    df_0 = df_0.applyfunc(simplify_term_dt).xreplace({dt: sample_period})
    df_1 = df_1.applyfunc(simplify_term_dt).xreplace({dt: sample_period})
    dF_0 = dF_0.applyfunc(simplify_term_dt).xreplace({dt: sample_period})
    dF_1 = dF_1.applyfunc(simplify_term_dt).xreplace({dt: sample_period})

    # create function from matrix
    df_0 = sympy.lambdify(x, df_0, modules='numpy')
    df_1 = sympy.lambdify(x, df_1, modules='numpy')
    dF_0 = sympy.lambdify(x, dF_0, modules='numpy')
    dF_1 = sympy.lambdify(x, dF_1, modules='numpy')

    # define measurement function h
    h = sympy.Matrix([
        px + wheelbase/2*sympy.cos(yaw_k),
        py + wheelbase/2*sympy.sin(yaw_k),
        dyaw_k,
        dv_k
    ])
    # define Jacobian H for measurement function h
    H = h.jacobian(x)


    # create function from matrix
    H_ = sympy.lambdify(x, H, modules='numpy')
    if wheelbase == 0.0:
        # Can just use H instead of both h and H
        # and H does not need to be a function as it is not state dependent
        h = None
        H = H_(0, 0, 0, 0, 0, 0) # H is not state dependent
    else:
        h_ = sympy.lambdify(x, h, modules='numpy')
        h = lambda state: h_(*state).reshape((-1, 1))
        H = lambda state: H_(*state)

    def f(state):
        if abs(state[4]) < yaw_rate_tolerance:
            df = df_0
        else:
            df = df_1
        return (state + df(*state)).reshape((-1, 1))

    def F(state):
        if abs(state[4]) < yaw_rate_tolerance:
            dF = dF_0
        else:
            dF = dF_1
        return np.eye(6) + dF(*state)

    return f, h, F, H


KalmanResult = collections.namedtuple(
    'KalmanResult', ['state_estimate',
                     'error_covariance',
                     'kalman_gain',
                     'predicted_state_estimate',
                     'predicted_error_covariance'])

class Kalman(object):
    def __init__(self, F, H, Q, R, f=None, h=None):
        """Initialize a object to perform Kalman filtering and smoothing.

        Parameters:
        F: State-transition model. A constant matrix of shape (n, n) or a
           function taking n arguments and returning a matrix of shape (n, n).
        H: Observation model. A constant matrix of shape (l, n) or a function
           taking n arguments and returning a matrix of shape (l, n).
        Q: Process noise covariance. A constant matrix of shape (n, n).
        R: Measurement noise covariance. A constant matrix of shape (l, l)

        f: Nonlinear state-transition model. A function taking n arguments and
           returning a matrix of shape (n, 1).
        h: Nonlinear observation model. A function taking l arguments and
           returning a matrix of shape (l, 1).

        where n is the number of states and l is the number of measurements.

        Note:
        If f/h is not None, then F/H must be supplied as a function as it
        represents the state-dependent Jacobian of f/h. If f/h is None, then F/H
        must be supplied as a constant matrix (linear function).
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        # if f/h is not None, F/H is the corresponding state dependent jacobian
        # and must be a function
        self.f = f
        self.h = h
        if f is not None:
            assert callable(f) and callable(F)
        if h is not None:
            assert callable(h) and callable(H)

    def estimate(self, x0, P0, z, y_steptol=None, progress=False):
        """Return the Kalman Filter estimate.

        Parameters:
        x0: Initial state estimate.
        P0: Initial error covariance.
        z: Measurements.
        y_steptol: Innovation step tolerance. Innovations exceeding the
                   corresponding step tolerance will have the associated
                   measurement ignored for a time step as the measurement is
                   assumed to be erroneous.
        progress: If True, display a progress bar while computing the estimate.
                  Defaults to False.
        """
        n = x0.shape[0] # state size
        l = z.shape[1] # measurement size
        N = z.shape[0] # number of samples

        x_hat_ = np.zeros((N, n, 1))    # predicted state estimate
        P_ = np.zeros((N, n, n))        # predicted error covariance
        x_hat = x_hat_.copy()           # (corrected) state estimate
        P = P_.copy()                   # (corrected) error covariance
        K = np.zeros((N, n, l))         # kalman gain

        x_hat[-1] = x0.copy().reshape((-1, 1))
        P[-1] = P0.copy()

        if y_steptol is None:
            y_steptol = np.zeros(n)
        else:
            y_steptol = np.asarray(y_steptol).astype(np.float).reshape((-1, 1))
            assert np.all(y_steptol >= 0.0)
        y_steptol[y_steptol == 0.0] = np.inf

        # turn measurements into a masked array if not already
        # as we later handle missing (masked) values
        if not isinstance(z, np.ma.MaskedArray):
            z = np.ma.array(z)

        for k in range(N):
            # time update
            if self.f is not None:
                x_hat_[k] = self.f(x_hat[k - 1])
                F = self.F(x_hat_[k])
            else:
                # if f is None, F is not state dependent
                # and we use the linear Kalman filter equation for time update
                F = self.F
                x_hat_[k] = F@x_hat[k - 1]
            P_[k] = F@P[k - 1]@F.T + self.Q

            # measurement update
            if self.h is not None:
                y = z[k].data.reshape((-1, 1)) - self.h(x_hat_[k])
                H = self.H(x_hat_[k])
            else:
                H = self.H
                y = z[k].data.reshape((-1, 1)) - H@x_hat_[k]

            # skip use of measurement if value is masked or
            # if value if far from the predicted value
            missing_mask = z[k].mask.reshape((-1, 1)) | (np.abs(y) > y_steptol)
            H[missing_mask.squeeze(), :] = 0

            S = H@P_[k]@H.T + self.R
            K[k] = np.linalg.solve(S.T, (P_[k]@H.T).T).T
            x_hat[k] = x_hat_[k] + K[k]@y
            P[k] = (np.eye(n) - K[k]@H)@P_[k]

            if progress:
                # print progress bar
                percent = int(np.ceil(k/(N + 1)*1000))
                s = '='*(percent//10) + ' '*(100 - percent//10)
                sys.stdout.write('\r')
                sys.stdout.write('[{}] {}%'.format(s, percent/10))
                sys.stdout.flush()

        return KalmanResult(
                state_estimate=x_hat,
                error_covariance=P,
                kalman_gain=K,
                predicted_state_estimate=x_hat_,
                predicted_error_covariance=P_)

    def smooth_estimate(self, result, progress=False):
        raise NotImplementedError

def plot_kalman_result(result, event=None, ax=None, **fig_kw):
    if ax is None:
        fig, ax = plt.subplots(3, 2, sharex=True, **fig_kw)
    else:
        fig = ax.get_figure()
    ax = ax.ravel()
    ax[0].axis('off')
    ax[1].axis('off')

    x = result.state_estimate
    P = result.error_covariance

    if event is None:
        event_time = np.arange(x.shape[0])
    else:
        event_time = event.bicycle.time
        T = np.diff(event.bicycle.time).mean()
        z = generate_measurement(event)

    color = sns.color_palette('tab10', 10)

    ax01 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2, fig=fig)
    ax01.plot(x[:, 0], x[:, 1],
              color=color[0], alpha=0.5,
              label='KF trajectory')
    if event is not None:
        index = ~z.mask.any(axis=1)
        ax01.scatter(x[index, 0],
                      x[index, 1],
                      s=15, marker='X',
                      color=color[0],
                      label='KF trajectory (valid measurement)')
        ax01.scatter(z[:, 0].compressed(),
                      z[:, 1].compressed(),
                      s=15, marker='X',
                      color=color[1], alpha=0.5,
                      label='trajectory (butter)')
    ax01.fill_between(x[:, 0].squeeze(),
                       x[:, 1].squeeze() + np.sqrt(P[:, 1, 1]),
                       x[:, 1].squeeze() - np.sqrt(P[:, 1, 1]),
                       color=color[0], alpha=0.2)
    ax01.legend()

    ax[2].plot(event_time, x[:, 2],
               color=color[0],
               label='KF yaw angle')
    if event is not None:
        ax[2].plot(event_time[1:],
                   scipy.integrate.cumtrapz(z[:, 2], dx=T) + np.pi,
                   color=color[3], alpha=0.5,
                   label='integrated yaw rate')
    ax[2].fill_between(event_time,
                       x[:, 2].squeeze() + np.sqrt(P[:, 2, 2]),
                       x[:, 2].squeeze() - np.sqrt(P[:, 2, 2]),
                       color=color[0], alpha=0.2)
    ax[2].legend()

    ax[3].plot(event_time, x[:, 4],
               color=color[0],
               label='KF yaw rate')
    if event is not None:
        ax[3].plot(event_time, z[:, 2],
                   color=color[1], alpha=0.5,
                   label='yaw rate')
    ax[3].fill_between(event_time,
                       x[:, 4].squeeze() + np.sqrt(P[:, 4, 4]),
                       x[:, 4].squeeze() - np.sqrt(P[:, 4, 4]),
                       color=color[0], alpha=0.2)
    ax[3].legend()

    ax[4].plot(event_time, x[:, 3],
               color=color[0],
               label='KF speed')
    ylim = ax[4].get_ylim()
    if event is not None:
        ax[4].plot(event_time, ff.moving_average(event.bicycle.speed, 55),
                   color=color[1], alpha=0.5,
                   label='speed')
        ax[4].plot(event_time[1:],
                   scipy.integrate.cumtrapz(z[:, 3], dx=T) + x[0, 3], #FIXME x0
                   color=color[3], alpha=0.5,
                   label='integrated accel')
        ax[4].plot(event_time, event.bicycle.speed,
                   color=color[1], alpha=0.2,
                   label='speed (raw)')
    ax[4].fill_between(event_time,
                       x[:, 3].squeeze() + np.sqrt(P[:, 3, 3]),
                       x[:, 3].squeeze() - np.sqrt(P[:, 3, 3]),
                       color=color[0], alpha=0.2)
    ax[4].set_ylim(ylim)
    ax[4].legend()

    ax[5].plot(event_time, x[:, 5],
               zorder=2,
               color=color[0], label='KF accel')
    ax[5].fill_between(event_time,
                       x[:, 5].squeeze() + np.sqrt(P[:, 5, 5]),
                       x[:, 5].squeeze() - np.sqrt(P[:, 5, 5]),
                       color=color[0], alpha=0.2)
    if event is not None:
        ax[5].plot(event_time, z[:, 3],
                   zorder=1,
                   color=color[1], alpha=0.5,
                   label='acceleration')
    ax[5].legend()
    return fig, ax


def plot_kalman_result_matrix(result_matrix, ax=None, **fig_kw):
    _, rows, cols = result_matrix.shape

    if ax is None:
        fig, ax = plt.subplots(rows, cols, sharex=True, **fig_kw)
    else:
        fig = ax.get_figure()

    color = sns.husl_palette(rows*cols, l=0.7)
    for i in range(rows):
        for j in range(cols):
            ax[i, j].plot(result_matrix[:, i, j],
                          color=color[i*cols + j])

    return fig, ax


# used in old path estimation (path.py)
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
