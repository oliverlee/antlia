#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import pickle
import matplotlib.pyplot as plt

from antlia import record
from plot import plot_timeseries


def get_metric(rec, start_index, stop_index, field, metric):
    indices = (rec.time > start_index) & (rec.time < stop_index)
    m = getattr(rec[indices][field], metric)
    return m()


if __name__ == '__main__':
    with open('config.p', 'rb') as f:
        cd = pickle.load(f)

    pat = os.path.join(os.path.dirname(__file__),
                       '../data/etrike/calibration/convbike/speed/*kph.csv')
    calfiles = glob.glob(pat)
    r = {}
    for f in calfiles:
        bn = os.path.basename(f)
        v = int(bn[:bn.index('kph')])
        r[v] = record.load_file(f, cd['convbike'])

        plot_timeseries(r[v])
    plt.show()
