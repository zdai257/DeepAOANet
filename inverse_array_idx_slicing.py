import os
from os.path import join
import argparse
import rospy
import math
from math import e
from math import *
import numpy as np
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
from pyargus import directionEstimation as de


parser = argparse.ArgumentParser(description='Specify ROS create_phaser params')
parser.add_argument('--slice', type=str, required=False, help='Specify num of slices for the IQ window')
parser.add_argument('--theta', type=str, required=False, help='Specify thata of impinging signal')
parser.add_argument('--inverse', type=str, required=False, help='Specify order of antenna array')
args = parser.parse_args()

if args.slice is not None:
    slice = float(args.slice)
else:
    slice = 8
if args.theta is not None:
    theta = float(args.theta)
else:
    theta = 0.
if args.inverse is not None:
    inverse = bool(args.inverse)
else:
    inverse = False


window_len = 32768
win_size = window_len//slice
alpha = 0.2
win_lst = range(0, window_len, win_size)

print("Inversing Array Indexing to Update New Correlation Matrix R: ", inverse)
print(win_lst)


def callback(msg):
    data = Float32MultiArray()

    # Format 'data' to be publish
    M0 = 4
    msg_dimx = MultiArrayDimension()
    msg_dimx.label = "height"
    msg_dimx.size = M0
    msg_dimx.stride = 2 * M0 * M0
    data.layout.dim.append(msg_dimx)

    msg_dimy = MultiArrayDimension()
    msg_dimy.label = "width"
    msg_dimy.size = M0
    msg_dimy.stride = 2 * M0
    data.layout.dim.append(msg_dimy)

    msg_dimz = MultiArrayDimension()
    msg_dimz.label = "imag"
    msg_dimz.size = 2
    msg_dimz.stride = 2
    data.layout.dim.append(msg_dimz)

    data.layout.data_offset = 0

    # Parse received 'msg'
    new_msg = Float32MultiArray()
    new_msg.layout = msg.layout

    M = msg.layout.dim[0].size
    N = msg.layout.dim[1].size
    len = M * N * 2


    new_msg.data = msg.data

    new_samples = np.asarray(new_msg.data).reshape(M, N, 2)
    data_real = new_samples[:, :, 0]
    data_imag = new_samples[:, :, 1]
    iq_samples = data_real + 1j * data_imag

    new_iq_samples = iq_samples

    # Get new R and publish; Inverse the order of iq_samples rows
    if inverse:
        for i in range(4):
            new_iq_samples[i] = iq_samples[3-i]

    for win_idx, win_val in enumerate(win_lst):
        win_samples = new_iq_samples[:, win_val:win_val + win_size]
        new_R = de.corr_matrix_estimate(win_samples.T, imp="fast")
        new_R_real = new_R.real
        new_R_imag = new_R.imag

        data_arr = np.append(new_R_real.reshape(M0, M0, 1), new_R_imag.reshape(M0, M0, 1), axis=2)
        data_lst = list(data_arr.ravel())
        data.data = data_lst
        pub_dict[win_idx].publish(data)


rospy.init_node('inverser', anonymous=True)

pub_dict = {}
for win_idx, win_val in enumerate(win_lst):
    pub = rospy.Publisher('/kerberos/R_inv' + str(win_idx), Float32MultiArray, queue_size=10)
    pub_dict[win_idx] = pub

topic = '/kerberos/iq_arr'
rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()
