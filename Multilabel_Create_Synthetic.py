import os
from os.path import join
import argparse
import rospy
import message_filters
import math
from math import e
from math import *
import numpy as np
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
from pyargus import directionEstimation as de


parser = argparse.ArgumentParser(description='Specify ROS Multilabel_Create_Synthetic params')
parser.add_argument('--theta1', type=str, required=True, help='Specify thata of impinging signal 1')
parser.add_argument('--theta2', type=str, required=True, help='Specify thata of impinging signal 2')
parser.add_argument('--IQgen', type=str, required=False, help='Specify whether to publish IQ data')
parser.add_argument('--mean', type=str, required=False, help='Specify MEAN of AWGN to IQ_samples')
parser.add_argument('--slice', type=str, required=False, help='Specify num of slices for the IQ window')
parser.add_argument('--inverse', type=str, required=False, help='Specify order of antenna array')
args = parser.parse_args()

if args.theta1 is not None:
    theta1 = float(args.theta1)
else:
    theta1 = -70.
if args.theta2 is not None:
    theta2 = float(args.theta2)
else:
    theta2 = -60.
if args.IQgen is not None:
    IQgen = True
else:
    IQgen = False
if args.mean is not None:
    gauss_mean = float(args.mean)
else:
    gauss_mean = 0.0
if args.slice is not None:
    slice = float(args.slice)
else:
    slice = 8
if args.inverse is not None:
    inverse = args.inverse
else:
    inverse = False


# Specify AWGN Sigma!
gauss_sigma = 1e-7

# Specify Angles that generate Phase Shift!
angle_dict = {'-4': -4, '-2': -2, '0': 0, '2': 2, '4': 4}
#angle_dict = {'0': 0,}

window_len = 32768
win_size = window_len//slice
alpha = 0.2
win_lst = range(0, window_len, win_size)

# Carrier Phase Difference
delta_phase = np.random.uniform(0, 2*pi, len(angle_dict.keys()))

IQamp_thres = 3e-3

print("Synthetic Angle Manipulators = ", angle_dict.keys())
print("Sigma of AWGN equals = ", gauss_sigma)
print("Slices = %d; Inverse = %s" % (slice, str(inverse)))
print("Delta_carrier_phase = ", delta_phase)


def sync_callback(msg1, msg2):
    data = Float32MultiArray()
    iq_data = Float32MultiArray()
    iq_data.layout = msg1.layout

    # Format 'data' to be publish
    M0 = 4
    msg_dimx = MultiArrayDimension()
    msg_dimx.label = "height"
    msg_dimx.size = M0
    msg_dimx.stride = M0 * M0 * 2 * slice
    data.layout.dim.append(msg_dimx)

    msg_dimy = MultiArrayDimension()
    msg_dimy.label = "width"
    msg_dimy.size = M0
    msg_dimy.stride = M0 * 2 * slice
    data.layout.dim.append(msg_dimy)

    msg_dimz = MultiArrayDimension()
    msg_dimz.label = "imag"
    msg_dimz.size = 2
    msg_dimz.stride = 2 * slice
    data.layout.dim.append(msg_dimz)

    msg_dims = MultiArrayDimension()
    msg_dims.label = "slice"
    msg_dims.size = slice
    msg_dims.stride = slice
    data.layout.dim.append(msg_dims)

    data.layout.data_offset = 0

    # Parse received 'msg'
    new_msg = Float32MultiArray()
    new_msg.layout = msg1.layout

    M = msg1.layout.dim[0].size
    N = msg1.layout.dim[1].size
    len = M * N * 2

    for idx2, (key2, angle_val2) in enumerate(angle_dict.items()):
        theta_rad2 = angle_val2 * pi / 180
        theta_angle2 = theta2 + angle_val2
        '''
        # Add AWGN
        noise2 = np.random.normal(gauss_mean, gauss_sigma, len)
        msg2.data += noise2
        '''
        # Form IQ2 as array
        new_samples2 = np.asarray(msg2.data).reshape(M, N, 2)
        iq_np2 = new_samples2[:, :, 0] + 1j * new_samples2[:, :, 1]

        # Introduce phase shift to different CHN of iq_samples HERE!
        for chn in range(0, 4):
            # Phase Shift
            phase2 = e ** (1j * 2 * pi * chn * alpha * sin(theta_rad2))
            iq_np2[chn] = iq_np2[chn] * phase2

        for idx, (key, angle_val) in enumerate(angle_dict.items()):

            theta_rad = angle_val * pi / 180
            theta_angle1 = theta1 + angle_val
            '''
            # Add AWGN
            noise1 = np.random.normal(gauss_mean, gauss_sigma, len)
            msg1.data += noise1
            '''
            if theta_angle1 != theta_angle2:
                # Form IQ1 as array
                new_samples1 = np.asarray(msg1.data).reshape(M, N, 2)
                iq_np1 = new_samples1[:, :, 0] + 1j * new_samples1[:, :, 1]

                # Allow reversing IQ1 only
                iq_tmp = iq_np1
                if inverse:
                    for i in range(4):
                        iq_np1[i] = iq_tmp[3 - i]

                # Introduce phase shift to different CHN of iq_samples HERE!
                for chn in range(0, 4):
                    # Phase Shift
                    phase = e ** (1j * 2 * pi * chn * alpha * sin(theta_rad))
                    iq_np1[chn] = iq_np1[chn] * phase

                # INSERT IQ-AMPLITUDE THRESHOLDS HERE!
                iq1_start = np.mean(np.sqrt(iq_np1[0, :100].real**2 + iq_np1[0, :100].imag**2))
                iq1_end = np.mean(np.sqrt(iq_np1[0, -100:].real ** 2 + iq_np1[0, -100:].imag ** 2))
                iq2_start = np.mean(np.sqrt(iq_np2[0, :100].real ** 2 + iq_np2[0, :100].imag ** 2))
                iq2_end = np.mean(np.sqrt(iq_np2[0, -100:].real ** 2 + iq_np2[0, -100:].imag ** 2))
                if iq1_start>IQamp_thres and iq1_end>IQamp_thres and iq2_start>IQamp_thres and iq2_end>IQamp_thres:

                    # Introduce Carrier Phase Randomizer HERE!
                    iq_np2_shifted = iq_np2 * e ** (1j * delta_phase[idx2])

                    # Superposition IQ1 & IQ2
                    new_iq_samples = iq_np1 + iq_np2_shifted

                    # Add AWGN only After the filtering!
                    new_iq_samples1 = new_iq_samples.real + np.random.normal(gauss_mean, gauss_sigma,
                                                                             new_iq_samples.real.shape)
                    new_iq_samples2 = new_iq_samples.imag + np.random.normal(gauss_mean, gauss_sigma,
                                                                             new_iq_samples.imag.shape)
                    new_iq_samples = new_iq_samples1 + 1j * new_iq_samples2
                    #print(new_iq_samples12.shape)

                    if IQgen:
                        iq_data.data = list(np.append(new_iq_samples.real.reshape(4, window_len, 1),
                                                      new_iq_samples.imag.reshape(4, window_len, 1), axis=2).ravel())
                        pub_iq.publish(iq_data)

                    R_slice = np.empty((M0, M0, 2, 0), dtype=np.float32)
                    for win_idx, win_val in enumerate(win_lst):
                        win_samples = new_iq_samples[:, win_val:win_val + win_size]
                        new_R = de.corr_matrix_estimate(win_samples.T, imp="fast")
                        new_R_real = new_R.real
                        new_R_imag = new_R.imag

                        data_arr = np.append(new_R_real.reshape((M0, M0, 1)), new_R_imag.reshape((M0, M0, 1)), axis=2)
                        R_slice = np.append(R_slice, data_arr.reshape((M0, M0, 2, 1)), axis=3)

                    data_lst = list(R_slice.ravel())
                    data.data = data_lst
                    pub_dict[idx2][idx].publish(data)


rospy.init_node('Multilabel_creator', anonymous=True)

pub_dict = {}
for idx2, (key2, angle_val2) in enumerate(angle_dict.items()):
    tmp_dict = {}
    for idx, (key, angle_val) in enumerate(angle_dict.items()):
        # Name theta in 3-digit degree
        theta1_name = str(int(theta1) + angle_val + 360)
        theta2_name = str(int(theta2) + angle_val2 + 360)
        if theta1_name != theta2_name:
            pub = rospy.Publisher('/kerberos/R_' + theta1_name + '_' + theta2_name, Float32MultiArray, queue_size=10)
            # pub = rospy.Publisher('/kerberos/R_multi', Float32MultiArray, queue_size=10)
            tmp_dict[idx] = pub

    pub_dict[idx2] = tmp_dict

if IQgen:
    pub_iq = rospy.Publisher('/kerberos/multilabel_iq_arr', Float32MultiArray, queue_size=1)


sub1 = message_filters.Subscriber('/IQ1', Float32MultiArray)
sub2 = message_filters.Subscriber('/IQ2', Float32MultiArray)

sync_client = message_filters.ApproximateTimeSynchronizer([sub1, sub2], queue_size=1, slop=.5, allow_headerless=True)
sync_client.registerCallback(sync_callback)

rospy.spin()
