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
parser.add_argument('--theta', type=str, required=True, help='Specify thata of impinging signal')
parser.add_argument('--inverse', type=bool, required=False, help='Specify order of antenna array')
args = parser.parse_args()
theta = float(args.theta)
if args.inverse is not None:
    inverse = args.inverse
else:
    inverse = False


alpha = 0.2
# Specify Angles that generate Phase Shift!
angle_dict = {'-4': -4, '-2': -2, '2': 2, '4': 4}

print("Synthetic Angle Manipulators = ", angle_dict.keys())


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

    for idx, (key, angle_val) in enumerate(angle_dict.items()):

        new_msg.data = msg.data

        new_samples = np.asarray(new_msg.data).reshape(M, N, 2)
        data_real = new_samples[:, :, 0]
        data_imag = new_samples[:, :, 1]
        iq_samples = data_real + 1j * data_imag

        new_iq_samples = iq_samples
        theta_deg = int(theta + angle_val)
        theta_rad = (theta + angle_val) * pi / 180

        # Get new R and publish
        # Introduce phase shift to iq_samples HERE!
        for chn in range(1, 4):

            # Phase Shift
            if inverse==True:
                phase = e ** (-1j * 2 * pi * chn * alpha * sin(theta_rad))
            else:
                phase = e ** (1j * 2 * pi * chn * alpha * sin(theta_rad))
            new_iq_samples[chn] = iq_samples[chn] * phase

        new_R = de.corr_matrix_estimate(new_iq_samples.T, imp="fast")
        new_R_real = new_R.real
        new_R_imag = new_R.imag

        data_arr = np.append(new_R_real.reshape(M0, M0, 1), new_R_imag.reshape(M0, M0, 1), axis=2)
        data_lst = list(data_arr.ravel())
        data.data = data_lst
        pub_dict[key].publish(data)


rospy.init_node('phaser_creator', anonymous=True)
#pub = rospy.Publisher('/kerberos/newR', Float32MultiArray, queue_size=10)
pub_dict = {}
for idx, (key, angle_val) in enumerate(angle_dict.items()):
    theta_deg = int(theta + angle_val)
    pub = rospy.Publisher('/kerberos/R_deg' + str(theta_deg), Float32MultiArray, queue_size=10)
    pub_dict[key] = pub

topic = '/kerberos/iq_arr'
rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()
