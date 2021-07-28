import os
from os.path import join
import argparse
import rospy
from math import *
import numpy as np
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
import pickle
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import keras


class DeepAOANet(object):
    def __init__(self):
        self.aoa = 0.
        self.data = None
        self.rdy_flag = False

        pkl_filename = "encoder_fc_tuple.pkl"
        with open(join('checkpoints', pkl_filename), 'rb') as a_file:
            encoder, model_4 = pickle.load(a_file)
        encoder.summary()
        model_4.summary()
        self.encoder, self.model_4 = encoder, model_4

        ### START QtApp #####
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow(title="AOA Compass")  # creates a window
        p = win.addPlot(title="Realtime plot")  # creates empty space for the plot in the window
        self.curve = p.plot()  # create an empty "plot" (a curve to plot)

        windowWidth = 500  # width of the window displaying the curve
        self.Xm = np.linspace(0, 0, windowWidth)  # create array that will contain the relevant time series
        self.ptr = -windowWidth  # set first x position


    def update(self):
        #global curve, ptr, Xm
        self.Xm[:-1] = self.Xm[1:]  # shift data in the temporal mean 1 sample left
        self.Xm[-1] = self.aoa  # vector containing the instantaneous values
        self.ptr += 1  # update x position for displaying the curve
        self.curve.setData(self.Xm)  # set the curve with this data
        self.curve.setPos(self.ptr, 0)  # set x position in the graph to 0
        QtGui.QApplication.processEvents()  # you MUST process the plot now

    def infer(self):
        out1 = self.encoder.predict(self.data)
        out2 = self.model_4.predict(out1)

        # Normalize output

        self.aoa = out2

    def data_ready(self):
        if self.rdy_flag:
            self.rdy_flag = False
            return True
        else:
            return False

    def callback(self, msg):
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

        # Buffer previous 'z' to make Time Series

        self.data = z
        self.rdy_flag = True


if __name__ == "__main__":
    AOA = DeepAOANet

    rospy.init_node('DeepAOAgui', anonymous=True)
    rospy.Subscriber('/kerberos/r', Float32MultiArray, AOA.callback)

    while not rospy.is_shutdown():
        if AOA.data_ready:
            # Inference
            print(AOA.aoa)
            # GUI Display
            AOA.update()

        rospy.spin()

    ### END QtApp ####
    pg.QtGui.QApplication.exec_()  # you MUST put this at the end
