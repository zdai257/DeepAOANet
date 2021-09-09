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
parser.add_argument('--slice', type=str, required=False, help='Specify num of slices for the IQ window')
parser.add_argument('--inverse', type=str, required=False, help='Specify order of antenna array')
args = parser.parse_args()
if args.sigma is not None:
    gauss_sigma = float(args.sigma)
else:
    gauss_sigma = 1e-5
if args.mean is not None:
    gauss_mean = float(args.mean)
else:
    gauss_mean = 0.0
if args.slice is not None:
    slice = float(args.slice)
else:
    slice = 8
if args.inverse is not None:
    inverse = bool(args.inverse)
else:
    inverse = False

# Specify AWGN Sigma!
#gauss_sigma_dict = {'1e_5': 1e-5, '5e_5': 5e-5, '1e_4': 1e-4, '5e_4': 5e-4}
#gauss_sigma_dict = {'1e_3': 1e-3, '2e_3': 2e-3, '3e_3': 3e-3, '4e_3': 4e-3, '5e_3': 5e-3}
gauss_sigma_dict = {'5e_4': 6e-4, '8e_4': 8e-4, '9e_4': 9e-4, '1e_3': 1e-3, '2e_3': 2e-3, '3e_3': 3e-3}

window_len = 32768
win_size = window_len//slice
alpha = 0.2
win_lst = range(0, window_len, win_size)

print("Sigma of AWGN equals = ", gauss_sigma_dict.keys())
print("Slicing window: ", win_lst)


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

        new_iq_samples = iq_samples

        if inverse:
            for i in range(4):
                new_iq_samples[i] = iq_samples[3 - i]

        # Get new R and publish
        # SLICE iq_samples HERE!
        for win_idx, win_val in enumerate(win_lst):
            win_samples = new_iq_samples[:, win_val:win_val + win_size]
            new_R = de.corr_matrix_estimate(win_samples.T, imp="fast")
            new_R_real = new_R.real
            new_R_imag = new_R.imag

            data_arr = np.append(new_R_real.reshape(M0, M0, 1), new_R_imag.reshape(M0, M0, 1), axis=2)
            data_lst = list(data_arr.ravel())
            data.data = data_lst
            pub_dict[key][win_idx].publish(data)



rospy.init_node('noisy_slicing_creator', anonymous=True)
#pub = rospy.Publisher('/kerberos/newR', Float32MultiArray, queue_size=10)
pub_dict = {}
for idx, (key, noi) in enumerate(gauss_sigma_dict.items()):
    tmp_dict = {}
    for win_idx, win_val in enumerate(win_lst):
        pub = rospy.Publisher('/kerberos/R_inv' + str(win_idx) + '_' + str(key), Float32MultiArray, queue_size=10)
        tmp_dict[win_idx] = pub
    pub_dict[key] = tmp_dict

topic = '/kerberos/iq_arr'
rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()
