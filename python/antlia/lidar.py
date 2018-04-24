# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation

from antlia.dtype import LIDAR_ANGLES


class LidarRecord(np.recarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def frame_index(self, time):
        """Get lidar frame(s) at time.

        Parameters:
        time: float or indexing function
        """
        if not callable(time):
            # return a one element LidarRecord
            return np.where(self.time >= time)[0][:1]

        return np.where(time(self.time))[0]

    def frame(self, time):
        index = self.frame_index(time)
        return self[index]

    def cartesian(self, xlim=None, ylim=None, rlim=None):
        rho = self.distance
        x = rho*np.cos(LIDAR_ANGLES)
        y = rho*np.sin(LIDAR_ANGLES)

        index = np.zeros(x.shape, dtype=bool)
        if xlim is not None:
            xmin, xmax = xlim
            index = index | (x < xmin) | (x > xmax)
        if ylim is not None:
            ymin, ymax = ylim
            index = index | (y < ymin) | (y > ymax)
        if rlim is not None:
            rmin, rmax = rlim
            index = index | (rho < rmin) | (rho > rmax)
        return (np.ma.masked_array(x, index),
                np.ma.masked_array(y, index))

    def animate(self, xlim=None, ylim=None, rlim=None,
                speedup=1, color=None, plot_kwargs={}, **kwargs):
        fig, ax = plt.subplots(**kwargs)

        plot_kwargs.setdefault('linestyle', ' ')
        plot_kwargs.setdefault('marker', '.')

        line, = ax.plot([], [], animated=True, **plot_kwargs)
        time_template = 'time = {:0.03f} s'
        if speedup != 1:
            time_template += ' (x{:0.1f})'.format(speedup)
        if speedup <= 0:
            raise ValueError

        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ax.set_xlabel('x position [meters]')
        ax.set_ylabel('y position [meters]')

        def init():
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def update(i):
            # get an array of one element so it's still a LidarRecord
            x, y = self[[i]].cartesian(xlim, ylim, rlim)
            line.set_data(x, y)

            if color is not None:
                line.set_color(color[i])

            time_text.set_text(time_template.format(self.time[i]))
            return line, time_text

        dt = np.diff(self.time)
        animator = matplotlib.animation.FuncAnimation(
                fig, update, frames=len(self), init_func=init,
                interval=int(dt.mean() * 1000 * 1/speedup),
                blit=True)

        return animator
