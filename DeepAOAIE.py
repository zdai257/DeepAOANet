import os
from os.path import join
import signal, sys
import argparse
import time
import rospy
from math import *
import numpy as np
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
import pickle
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from util.GUItest import Window
from pyargus import directionEstimation as de

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

import keras
from keras import backend as K

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# K.set_learning_phase(0)

parser = argparse.ArgumentParser(description='Specify DeepAOAIE params')
parser.add_argument('--model', type=str, required=False, help='Specify Model Type FC or CNN')
args = parser.parse_args()

if args.model is not None:
    model_type = str(args.model)
else:
    model_type = 'FC'


class DeepAOAIE(object):
    def __init__(self, model_name='FC', cast2image=True, IsMoveAxis=False):

        #self.aoa = None
        self.aoa_is_signal = False
        self.num_of_signal = 0
        self.theta1, self.theta2 = None, None
        self.ymin, self.ymax = -74, 74

        self.label_angls = np.arange(-74, 76, 2)
        self.input_data = None
        self.rdy_flag = False
        self.M = 4
        self.N = 32768
        self.IQamp_thres = 3e-3
        self.win_size = int(self.N / 8)
        self.win_lst = range(0, self.N, self.win_size)
        self.cast2image = cast2image
        self.IsMoveAxis = IsMoveAxis

        self.model_name = model_name
        self.timing = np.empty(0)

        if self.model_name == 'FC':
            self.pkl_filename = 'model_cr01'
        elif self.model_name == 'CNN':
            self.pkl_filename = 'model_cr11'
        else:
            raise ValueError('No such model!')

        '''
        with open(join('checkpoints', self.pkl_filename), 'rb') as a_file:
            model = pickle.load(a_file)
        '''
        model = keras.models.load_model(join('checkpoints', self.pkl_filename + '.h5'))
        model.summary()

        self.model = model
        self.sess = K.get_session()


    def infer(self):

        if self.input_data is not None:
            start_t = time.time()
            # Inference
            print("\nStart Infer")

            pred = self.sess.run([self.model.outputs],
                                 feed_dict={self.model.inputs[0]: self.input_data})

            elapsed_t = time.time() - start_t
            print("Inference Latency = %.4f" % elapsed_t)
            self.timing = np.append(self.timing, elapsed_t)

            # Output: a List of 3 np.array, val of which equals array[0, 0]
            print(pred)
            # Normalize output
            self.num_of_signal = pred[0][0][0, 0].round()

            self.theta1 = pred[0][1][0, 0] * (self.ymax - self.ymin) + self.ymin
            self.theta2 = pred[0][2][0, 0] * (self.ymax - self.ymin) + self.ymin
            print(self.num_of_signal)
            print(self.theta1, self.theta2)

        #else:
        #    print("Detect Noise!")


    def data_ready(self):
        if self.rdy_flag:
            ret = True
        else:
            ret = False

        self.reset_flag()
        return ret

    def set_flag(self):
        self.rdy_flag = True

    def reset_flag(self):
        self.rdy_flag = False

    def callback(self, msg):

        iq_data = np.asarray(msg.data).reshape(self.M, self.N, 2)
        iq_np = iq_data[:, :, 0] + 1j * iq_data[:, :, 1]

        # Threshold-based Noise Filter!
        iq_start = np.mean(np.sqrt(iq_np[0, :100].real ** 2 + iq_np[0, :100].imag ** 2))
        iq_end = np.mean(np.sqrt(iq_np[0, -100:].real ** 2 + iq_np[0, -100:].imag ** 2))

        if iq_start > self.IQamp_thres and iq_end > self.IQamp_thres:
            # Create 8-channel R
            R_slice = np.empty((self.M, self.M, 2, 0), dtype=np.float32)
            for win_idx, win_val in enumerate(self.win_lst):
                win_samples = iq_np[:, win_val:win_val + self.win_size]
                new_R = de.corr_matrix_estimate(win_samples.T, imp="fast")
                new_R_real = new_R.real
                new_R_imag = new_R.imag

                data_arr = np.append(new_R_real.reshape((self.M, self.M, 1)), new_R_imag.reshape((self.M, self.M, 1)), axis=2)
                R_slice = np.append(R_slice, data_arr.reshape((self.M, self.M, 2, 1)), axis=3)

            if self.cast2image:
                filtered_data = np.zeros((self.M, self.M, 8))
                for i in range(self.M):
                    for j in range(self.M):
                        if i <= j:
                            filtered_data[i, j, :] = R_slice[i, j, 0, :]
                        else:
                            filtered_data[i, j, :] = R_slice[i, j, 1, :]
                if self.IsMoveAxis:
                    filtered_data = np.moveaxis(filtered_data, -1, 0)

            else:
                filtered_data = np.zeros((10, 2, 8))
                k = 0
                for i in range(4):
                    for j in range(4):
                        if i <= j:
                            filtered_data[k, :, :] = R_slice[i, j, :, :]
                            k += 1
                if self.IsMoveAxis:
                    filtered_data = np.moveaxis(filtered_data, -1, 0)

            # Input Vec
            b = filtered_data.reshape((1, -1), order='C')

            # Normalization
            with open(join('checkpoints', 'StandardScaler-originaxis.pkl'), 'rb') as a_file:
                sscaler = pickle.load(a_file)

            self.input_data = sscaler.transform(b).reshape((1, -1))

            if self.model_name == 'CNN':
                if self.IsMoveAxis:
                    self.input_data = self.input_data.reshape((1, 8, 4, 4))
                    self.input_data = np.moveaxis(self.input_data, 1, -1)
                else:
                    self.input_data = self.input_data.reshape((1, 4, 4, 8))

        else:
            self.input_data = None
            self.theta1, self.theta2 = None, None
            self.num_of_signal = -1

        self.set_flag()


if __name__ == "__main__":

    # PyQt5 Program fixed writing
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda *a: app.quit())
    app.startTimer(200)

    # Instantiate and display the window bound to the drawing control
    AOAie = DeepAOAIE(model_name=model_type, IsMoveAxis=False)

    # Window
    win = pg.GraphicsWindow(title="AOA Spatial Power Spectrum")
    #pg.setConfigOption('background', 'w')
    #pg.setConfigOption('foreground', 'k')

    p = win.addPlot(title="Real-Time AOA Spatial Power Spectrum")  # creates empty space for the plot in the window

    p.setYRange(-0.1, 1.1, padding=0)

    envelope = p.plot(pen='k', name='PDF')  # create an empty "plot" (a curve to plot)
    #envelope.setScale(0.1)

    # Set X Axis
    p.setLabel('bottom', "Angles (degree)")

    ticks = [list(zip(range(-60, 1740, 200), ('-80', '-60', '-40', '-20', '0', '20', '40', '60', '80')))]

    xax = p.getAxis('bottom')
    xax.setTicks(ticks)

    vb = p.getViewBox()
    #vb.setForegroundColor((255, 255, 255))
    vb.setBackgroundColor((255, 255, 255))
    
    rospy.init_node('DeepAOAIE', anonymous=True)
    rospy.Subscriber('/kerberos/iq_arr', Float32MultiArray, AOAie.callback)

    
    # Realtime data plot. Each time this function is called, the data display is updated
    def update():
        global envelope, AOAie

        Xm = np.random.normal(loc=0.05, scale=0.005, size=1481)


        if AOAie.num_of_signal == 0:
            aoa = round(AOAie.theta1, 1)
            aoa_idx = int((aoa - AOAie.ymin) / 0.1)
            Xm[aoa_idx] = 1.


        elif AOAie.num_of_signal == 1:
            aoa1 = round(AOAie.theta1, 1)
            aoa2 = round(AOAie.theta2, 1)
            aoa1_idx = int((aoa1 - AOAie.ymin) / 0.1)
            aoa2_idx = int((aoa2 - AOAie.ymin) / 0.1)
            Xm[aoa1_idx] = 1.
            Xm[aoa2_idx] = 1.
        else:
            Xm = np.random.normal(loc=0.1, scale=0.01, size=1481)

        envelope.setData(Xm)  # set the curve with this data

        QApplication.processEvents()  # you MUST process the plot now


    while not rospy.is_shutdown():
        '''
        if AOAie.model_name == 'FC':
            AOAie.input_data = np.random.normal(loc=0., scale=0.2, size=(1, 128))
        elif AOAie.model_name == 'CNN':
            AOAie.input_data = np.random.normal(loc=0., scale=0.2, size=(1, 4, 4, 8))
        '''
        if AOAie.data_ready():
            AOAie.infer()

            # Saving Elapsed Times
            #np.save(join('doc', 'Timing_' + AOAie.model_name + '.npy'), AOAie.timing)
            
            # GUI Display
            update()



    # rospy.spin()

    ### END QtApp ####
    QApplication.exec_()  # you MUST put this at the end
    # PyQt5 Program fixed writing
    app.exec_()
    sys.exit(app.exec_())

