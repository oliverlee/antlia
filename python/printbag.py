#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert a rosbag file to legacy lidar binary format.
"""

"""LIDAR datatype format is:
    (
        timestamp (long),
        flag (bool saved as int),
        accelerometer[3] (double),
        gps[3] (double),
        distance[LIDAR_NUM_ANGLES] (long),
    )

    'int' and 'long' are the same size on the raspberry pi (32 bits).
"""
import sys
import logging

# suppress logging warnings due to rospy
logging.basicConfig(filename='/dev/null')
import rosbag

def print_bag(bag):
    topics = ['/scan', '/flagbutton_pressed']
    for message in bag.read_messages(topics=topics):
        print(message)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(('Usage: {} <rosbag> [<outfile>] \n\n'
               'Print contents of rosbag file. If <outfile> is provided, \n'
               'write contents of rosbag file to <outfile> in the legacy \n'
               'lidar binary format.').format(__file__))
        sys.exit(1)

    outfile = None
    filename = sys.argv[1]

    if len(sys.argv) == 3:
        # convert to outfile if provided
        outfile = sys.argv[2]

    with rosbag.Bag(filename) as bag:
        print_bag(bag)

    sys.exit()

