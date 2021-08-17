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

os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#K.set_learning_phase(0)


class DeepAOANet(object):
    def __init__(self):

        self.aoa = None
        self.aoa_is_signal = False
        self.data = None
        self.rdy_flag = False
        self.inp = {0: None, -1: None, -2: None}

        pkl_filename = "LSTM-AE-4FC_X_105.pkl"
        with open(join('checkpoints', pkl_filename), 'rb') as a_file:
            model_encoder_fc = pickle.load(a_file)
        model_encoder_fc.summary()

        self.model = model_encoder_fc
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

        if any(val is None for val in self.inp.values()):
            self.aoa = None
            print("t < 3 Not ready for Inference.")
        else:
            new_inp = np.zeros((1, 3, 20), dtype='float32')
            new_inp[0, 0] = self.inp[-2]
            new_inp[0, 1] = self.inp[-1]
            new_inp[0, 2] = self.inp[0]

            out = self.sess.run([self.model.outputs],
                                feed_dict={self.model.inputs[0]: new_inp})

            # print(out.shape)
            # Normalize output

            self.aoa = out

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
        msg_len = M * N * dimz

        R = np.asarray(msg.data, dtype='float32').reshape(M, M, dimz)

        b_lst = []
        for i in range(M):
            for j in range(N):
                if i <= j:
                    b_lst.append(R[i, j, 0])
                    b_lst.append(R[i, j, 1])

        b = np.array(b_lst)
        z = b / np.linalg.norm(b)

        self.data = z

        # Filter out Noises!
        field_data2 = np.linalg.norm(R[0, 1, :])
        if field_data2 > 1e-05:
            self.aoa_is_signal = True
        else:
            self.aoa_is_signal = False
        
        # Circular buffer
        self.inp[-2] = self.inp[-1]
        self.inp[-1] = self.inp[0]
        self.inp[0] = self.data

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
    rospy.Subscriber('/kerberos/r', Float32MultiArray, AOA.callback)


    # Realtime data plot. Each time this function is called, the data display is updated
    def update():
        global envelope, AOA

        #Xm = np.random.normal(size=141)
        Xm = np.zeros(141, dtype='float32')

        xm = round(AOA.aoa[0][0][0, 0] * 140 - 70)
        Xm[xm] = 1.

        envelope.setData(Xm)  # set the curve with this data

        QApplication.processEvents()  # you MUST process the plot now


    while not rospy.is_shutdown():
        if AOA.data_ready():
            start_t = time.time()
            # Inference
            print("\nStart Infer")
            AOA.infer()
            if AOA.aoa is not None:

                print("AOA = %.5f" % (AOA.aoa[0][0]))

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

