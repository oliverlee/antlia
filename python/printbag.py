#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print a rosbag file.
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
        #topics = ['/tf', '/scan']
        topics = ['/scan', '/flagbutton_pressed']

    for message in bag.read_messages(topics=topics):
        print(message)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(('Usage: {} <rosbag> \n\n'
               'Print contents of rosbag file.'
               ).format(__file__))
        sys.exit(1)

    outfile = None
    filename = sys.argv[1]

    with rosbag.Bag(filename) as bag:
        print_bag(bag)

    sys.exit()
