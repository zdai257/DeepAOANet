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
from GUItest import Window
import keras
from keras import backend as K

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#K.set_learning_phase(0)


class DeepAOANet(object):
    def __init__(self, pkl_filename="model_fc0.pkl"):

        self.aoa = None
        self.aoa_is_signal = False
        self.label_angls = np.arange(-74, 76, 2)
        self.data = None
        self.rdy_flag = False
        self.inp = {0: None, -1: None, -2: None}  # buffer for Time Series
        self.pkl_filename = pkl_filename

        with open(join('checkpoints', self.pkl_filename), 'rb') as a_file:
            model_fc0 = pickle.load(a_file)
        model_fc0.summary()

        self.model = model_fc0
        self.sess = K.get_session()
        '''
        ### START QtApp #####
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow(title="AOA Compass")  # creates a window
        p = win.addPlot(title="Realtime plot")  # creates empty space for the plot in the window
        self.curve = p.plot()  # create an empty "plot" (a curve to plot)

        windowWidth = 500  # width of the window displaying the curve
        self.Xm = np.linspace(0, 0, windowWidth)  # create array that will contain the relevant time series
        self.ptr = -windowWidth  # set first x position
        '''

    '''
    def update(self):
        #global curve, ptr, Xm
        self.Xm[:-1] = self.Xm[1:]  # shift data in the temporal mean 1 sample left
        self.Xm[-1] = self.aoa  # vector containing the instantaneous values
        self.ptr += 1  # update x position for displaying the curve
        self.curve.setData(self.Xm)  # set the curve with this data
        self.curve.setPos(self.ptr, 0)  # set x position in the graph to 0
        QtGui.QApplication.processEvents()  # you MUST process the plot now
    '''

    def infer(self):

        pred = self.sess.run([self.model.outputs],
                            feed_dict={self.model.inputs[0]: self.data})

        print(pred)
        # Normalize output
        pred_round = pred[0][0].round()
        y_idx = np.argmax(pred_round, axis=-1)

        self.aoa = self.label_angls[y_idx]

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
        #print("\ncallback!")
        # Parse received 'msg'
        M = msg.layout.dim[0].size
        N = msg.layout.dim[1].size
        dimz = 2
        CHN = msg.layout.dim[3].size
        msg_len = M * N * dimz * CHN

        msg_data = np.asarray(msg.data, dtype='float32').reshape(M, M, dimz, CHN)

        # Prepare array in the right sequence
        filtered_data = np.zeros((10, dimz, CHN))
        k = 0
        for i in range(M):
            for j in range(M):
                if i <= j:
                    filtered_data[k, :, :] = msg_data[i, j, :, :]
                    k += 1

        filtered_data = np.moveaxis(filtered_data, -1, 0)
        R = filtered_data.reshape((1, -1), order='C')

        # Filter out Noises!
        '''
        if field_data2 > 1e-05:
            self.aoa_is_signal = True
        else:
            self.aoa_is_signal = False
        '''
        
        # Normalization
        xmax = 0.5676375
        xmin = -0.46326497
        R_std = 2 * (R - xmin)/(xmax-xmin) - 1

        self.data = R_std
        self.set_flag()


if __name__ == "__main__":

    # PyQt5 Program fixed writing
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda *a: app.quit())
    app.startTimer(200)

    # Instantiate and display the window bound to the drawing control
    AOA = DeepAOANet()

    # Window
    win = pg.GraphicsWindow(title="AOA Spatial Power Spectrum")
    p = win.addPlot(title="Realtime plot")  # creates empty space for the plot in the window
    envelope = p.plot()  # create an empty "plot" (a curve to plot)

    rospy.init_node('DeepAOAgui', anonymous=True)
    rospy.Subscriber('/kerberos/R_310_0e_4', Float32MultiArray, AOA.callback)


    # Realtime data plot. Each time this function is called, the data display is updated
    def update():
        global envelope, AOA

        #Xm = np.random.normal(size=141)
        Xm = np.zeros(181, dtype='float32')

        #xm = round(AOA.aoa[0][0][0, 0] * 140 - 70)
        Xm[90 + AOA.aoa] = 1.

        envelope.setData(Xm)  # set the curve with this data

        QApplication.processEvents()  # you MUST process the plot now


    while not rospy.is_shutdown():
        if AOA.data_ready():
            start_t = time.time()
            # Inference
            print("\nStart Infer")
            AOA.infer()
            if AOA.aoa is not None:

                print("AOA = %d deg" % AOA.aoa)

                # GUI Display
                #if AOA.aoa_is_signal:
                update()

            elapsed_t = time.time() - start_t
            print("Inference Latency = %.4f" % elapsed_t)


    #rospy.spin()

    ### END QtApp ####
    QApplication.exec_()  # you MUST put this at the end
    # PyQt5 Program fixed writing
    app.exec_()
    sys.exit(app.exec_())

