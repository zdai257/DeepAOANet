import os
from os.path import join
import rosbag
import numpy as np

ROSBAG_PATH = join('data_1606', 'deg_0.bag')

bag = rosbag.Bag(ROSBAG_PATH, 'r')

for topic in ['/kerberos/iq_arr']:
    for subtopic, msg, t in bag.read_messages(topic):
        msgString = str(msg)
        msgList = str.split(msgString, '\n')
        for nameValuePair in msgList:
            print(nameValuePair)

        break
