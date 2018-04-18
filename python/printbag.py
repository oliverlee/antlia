#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert a rosbag file to legacy lidar binary format LIDAR_CONVERTED_DTYPE.
"""

import sys
import logging

import numpy as np

# suppress logging warnings due to rospy
logging.basicConfig(filename='/dev/null')
import rosbag

from antlia.dtype import LIDAR_CONVERTED_DTYPE

def print_bag(bag, topics=None):
    if topics is None:
        topics = ['/scan', '/flagbutton_pressed']

    for message in bag.read_messages(topics=topics):
        print(message)


def convert_bag(bag):
    topics = ['/scan', '/flagbutton_pressed']

    # assume button always starts as false
    button_state = False

    last_timestamp = 0
    record = []
    for topic, msg, t in bag.read_messages(topics=topics):

        # check that messages are monotonically increasing in time
        timestamp = msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs
        assert_msg = ('messages are not monotonically increasing in time\n'
                      'last_timestamp: {}\n'
                      'timestamp: {}\n').format(last_timestamp, timestamp)
        assert timestamp >= last_timestamp, assert_msg

        if topic == '/flagbutton_pressed':
            # use zero-order hold for button state
            button_state = msg.data
        elif topic == '/scan':
            # write datatype object for every lidar scan
            datum = np.recarray((1,), dtype=LIDAR_CONVERTED_DTYPE)

            # imu accelerometer and gps is ignored
            datum.time = timestamp
            datum.sync = button_state
            datum.distance = msg.ranges

            record.append(datum)
        else:
            raise KeyError('Unhandled topic: {}'.format(topic))

    return np.concatenate(record)


if __name__ == '__main__':
    OUTFILE_EXT = '.pkl.gz'

    if len(sys.argv) < 2:
        print(('Usage: {} <rosbag> [<outfile>] \n\n'
               'Print contents of rosbag file. If <outfile> is provided, \n'
               'write contents of rosbag file to <outfile> in the legacy \n'
               'lidar binary format LIDAR_CONVERTED_DTYPE.\n'
               'Output files will written with the extension \'{}\'.'
               ).format(__file__, OUTFILE_EXT))
        sys.exit(1)

    outfile = None
    filename = sys.argv[1]

    with rosbag.Bag(filename) as bag:
        if len(sys.argv) >= 3:
            import pickle
            import gzip

            # convert to outfile if provided
            outfile = sys.argv[2]
            record = convert_bag(bag)

            if not outfile.endswith(OUTFILE_EXT):
                outfile += OUTFILE_EXT
            pickle.dump(record, gzip.open(outfile, 'wb'))
        else:
            print_bag(bag)

    sys.exit()

