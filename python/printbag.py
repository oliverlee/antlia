#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print a rosbag file.
"""
import sys
import logging

# suppress logging warnings due to rospy
logging.basicConfig(filename='/dev/null')
import rosbag


def print_bag(bag, topics=None):
    for message in bag.read_messages(topics=topics):
        print(message)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(('Usage: {} [topics] <rosbag> \n\n'
               'topics:\tcomma-separated list of topics\n\n'
               'Print contents of rosbag file. If topics is not provided, \n'
               'all topics are printed\n'
               ).format(__file__))
        sys.exit(1)

    topics = None
    if len(sys.argv) == 3:
        topics = [t.strip() for t in sys.argv[1].split(',')]
        filename = sys.argv[2]
    else:
        filename = sys.argv[1]

    with rosbag.Bag(filename) as bag:
        print_bag(bag, topics)

    sys.exit()
