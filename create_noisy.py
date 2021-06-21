import os
from os.path import join
import argparse
import rospy
from math import *
import numpy as np
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
from pyargus import directionEstimation as de


parser = argparse.ArgumentParser(description='Specify ROS create_noisy params')
parser.add_argument('--sigma', type=str, required=False, help='Specify SIGMA of AWGN to IQ_samples')
parser.add_argument('--mean', type=str, required=False, help='Specify MEAN of AWGN to IQ_samples')
args = parser.parse_args()
if args.sigma is not None:
    gauss_sigma = float(args.sigma)
else:
    gauss_sigma = 1e-5
if args.mean is not None:
    gauss_mean = float(args.mean)
else:
    gauss_mean = 0.0


gauss_sigma_dict = {'1e_5': 1e-5, '5e_5': 5e-5, '1e_4': 1e-4, '5e_4': 5e-4}
print("Sigma of AWGN equals = ", gauss_sigma_dict.keys())


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

    for idx, (key, noise_sigma) in enumerate(gauss_sigma_dict.items()):
        noise = np.random.normal(gauss_mean, noise_sigma, len)
        new_msg.data = msg.data + noise

        new_samples = np.asarray(new_msg.data).reshape(M, N, 2)
        data_real = new_samples[:, :, 0]
        data_imag = new_samples[:, :, 1]
        iq_samples = data_real + 1j * data_imag

        # Get new R and publish
        new_R = de.corr_matrix_estimate(iq_samples.T, imp="fast")
        new_R_real = new_R.real
        new_R_imag = new_R.imag

        data_arr = np.append(new_R_real.reshape(M0, M0, 1), new_R_imag.reshape(M0, M0, 1), axis=2)
        data_lst = list(data_arr.ravel())
        data.data = data_lst
        pub_dict[key].publish(data)


rospy.init_node('noisy_creator', anonymous=True)
#pub = rospy.Publisher('/kerberos/newR', Float32MultiArray, queue_size=10)
pub_dict = {}
for idx, (key, noi) in enumerate(gauss_sigma_dict.items()):
    pub = rospy.Publisher('/kerberos/R' + str(key), Float32MultiArray, queue_size=10)
    pub_dict[key] = pub

topic = '/kerberos/iq_arr'
rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()
