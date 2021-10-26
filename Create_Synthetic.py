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
parser.add_argument('--A', type=str, required=False, help='Specify Amplitude of signal scalar')
parser.add_argument('--mean', type=str, required=False, help='Specify MEAN of AWGN to IQ_samples')
parser.add_argument('--slice', type=str, required=False, help='Specify num of slices for the IQ window')
parser.add_argument('--inverse', type=str, required=False, help='Specify order of antenna array')
args = parser.parse_args()

theta = float(args.theta)
if args.A is not None:
    A = float(args.A)
else:
    A = 1.0
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
gauss_sigma_dict = {'0e_4': 0e-4, '5e_4': 5e-4, '7e_4': 7e-4, '9e_4': 9e-4}
#gauss_sigma_dict = {'1e_1': 1e-1, '1e_2': 1e-2, '1e_3': 1e-3, '1e_4': 1e-4, '1e_5': 1e-5}
#gauss_sigma_dict = {'18db': 0.063, '16db': 0.0398, '14db': 0.0251, '12db': 0.0158}

# Specify Angles that generate Phase Shift!
angle_dict = {'-4': -4, '-2': -2, '0': 0, '2': 2, '4': 4}
#angle_dict = {'0': 0}

window_len = 32768
win_size = window_len//slice
alpha = 0.2
win_lst = range(0, window_len, win_size)

IQamp_thres = 3e-3

print("Synthetic Angle Manipulators = ", angle_dict.keys())
print("Sigma of AWGN equals = ", gauss_sigma_dict.keys())
print("Slices = %d; Inverse = %s" % (slice, str(inverse)))


def callback(msg):
    data = Float32MultiArray()
    iq_data = Float32MultiArray()

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
    new_msg.layout = msg.layout

    M = msg.layout.dim[0].size
    N = msg.layout.dim[1].size
    len = M * N * 2

    for idx_sigma, (sigma_name, noise_sigma) in enumerate(gauss_sigma_dict.items()):
        # Add AWGN
        noise = np.random.normal(gauss_mean, noise_sigma, len)
        new_msg.data = msg.data + noise

        iq_np0 = np.asarray(msg.data).reshape((M, N, 2))
        iq_np = iq_np0[:, :, 0] + 1j * iq_np0[:, :, 1]

        # INSERT IQ-AMPLITUDE THRESHOLDS HERE!
        iq1_start = np.mean(np.sqrt(iq_np[0, :100].real ** 2 + iq_np[0, :100].imag ** 2))
        iq1_end = np.mean(np.sqrt(iq_np[0, -100:].real ** 2 + iq_np[0, -100:].imag ** 2))
        if iq1_start > IQamp_thres and iq1_end > IQamp_thres:

            for idx, (key, angle_val) in enumerate(angle_dict.items()):

                new_samples = np.asarray(new_msg.data).reshape(M, N, 2)  #reshape(())
                data_real = new_samples[:, :, 0]
                data_imag = new_samples[:, :, 1]
                iq_samples = data_real + 1j * data_imag

                new_iq_samples = iq_samples

                theta_deg = int(theta + angle_val)
                # Should be Delta_theta HERE!
                theta_rad = angle_val * pi / 180

                if inverse:
                    for i in range(4):
                        new_iq_samples[i] = iq_samples[3 - i]

                new_iq_samples0 = new_iq_samples

                # Get new R and publish
                # Introduce phase shift to iq_samples HERE!
                for chn in range(0, 4):
                    # Phase Shift
                    phase = e ** (1j * 2 * pi * chn * alpha * sin(theta_rad))

                    new_iq_samples[chn] = new_iq_samples0[chn] * A * phase

                '''
                # Publish synthetic iq_data
                if angle_val == 0 and noise_sigma == 0. and A != 1:
                    angle_jump = 45  # Randomize it
                    syn_iq_samples = np.zeros(new_iq_samples0.shape)
                    # Superposition IQ samples
                    for chn in range(0, 4):
                        # Phase Shift
                        phase = e ** (1j * 2 * pi * chn * alpha * sin(angle_jump*pi/180))
                        # Adding old and new IQ is WRONG!!
                        syn_iq_samples[chn] = new_iq_samples0[chn] * (1. + A * phase)

                    iq_data.data = list(np.append(syn_iq_samples.real.reshape(4,window_len,1), syn_iq_samples.imag.reshape(4,window_len,1), axis=2).ravel())
                    pub_iq.publish(iq_data)
                '''

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
                pub_dict[sigma_name][idx].publish(data)



rospy.init_node('phaser_creator', anonymous=True)
#pub = rospy.Publisher('/kerberos/newR', Float32MultiArray, queue_size=10)
pub_dict = {}
for idx_sigma, (sigma_name, noise_sigma) in enumerate(gauss_sigma_dict.items()):
    tmp_dict = {}
    for idx, (key, angle_val) in enumerate(angle_dict.items()):
        theta_deg = int(theta + angle_val)
        # Name theta in 3-digit degree
        theta_name = str(theta_deg + 360)

        pub = rospy.Publisher('/kerberos/R_' + theta_name + '_' + sigma_name, Float32MultiArray, queue_size=100)
        tmp_dict[idx] = pub
    pub_dict[sigma_name] = tmp_dict

if A != 1:
    pub_iq = rospy.Publisher('/kerberos/syn_iq_arr', Float32MultiArray, queue_size=1)

topic = '/kerberos/iq_arr'
rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()
