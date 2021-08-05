from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import numpy as np
import math
import pyqtgraph as pq


class Window(QWidget):
    def __init__(self, data0=None):
        super().__init__()
        self.data2 = data0
        # Set the size
        self.resize(900, 600)
        # Add PlotWidget control
        self.plotWidget_ted = PlotWidget(self)
        # Set the size and relative position of the control
        self.plotWidget_ted.setGeometry(QtCore.QRect(25, 25, 850, 550))

        # Copy the data in the mode1 code
        self.data1 = np.zeros(141)

        self.curve1 = self.plotWidget_ted.plot(self.data1, name="mode1")

        # Set timer
        self.timer = pq.QtCore.QTimer()
        # Timer signal binding update_data function
        self.timer.timeout.connect(self.update_data)
        # The timer interval is 50ms, which can be understood as refreshing data once in 50ms
        self.timer.start(50)

    # Data shift left
    def update_data(self):
        if self.data2 is not None:
            new_data = self.data2
        else:
            new_data = np.zeros(141, dtype='float32')
            for i in range(141):
                mu = - math.log(abs(i - 71) / 100 + 0.5)
                sigma = 0.1
                new_data[i] = np.random.normal(mu, sigma)

        # Data is filled into the drawing curve
        self.curve1.setData(new_data)

    def new_data(self, data):
        self.data2 = data


if __name__ == '__main__':
    import sys
    # PyQt5 Program fixed writing
    app = QApplication(sys.argv)

    # Instantiate and display the window bound to the drawing control
    window = Window()
    window.setWindowTitle('AOA Power Spectrum')
    window.show()

    #window.new_data(np.ones(141))

    # PyQt5 Program fixed writing
    sys.exit(app.exec())