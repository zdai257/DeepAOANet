import os
import numpy as np
from gps_class import GPSVis
from threading import Thread
import socket
import struct

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


class SockFlag:
    def __init__(self):
        self.flag = False
        self.msg = None
        self.HOST = HOST  # Standard loopback interface address (localhost)
        self.PORT = PORT  # Port to listen on (non-privileged ports are > 1023)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.HOST, self.PORT))
        self.s.listen()
        self.connected = False

    def reconnect(self):
        self.flag = False
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.HOST, self.PORT))
        self.s.listen()
        self.connected = False

    def connect(self):
        self.conn, self.addr = self.s.accept()
        print('Connected by', self.addr)
        self.connected = True

    def loop(self):
        data = self.conn.recv(1024)  # 1024
        return data

    def Process(self, data):
        '''
        Raise Flag for main thread to Process
        '''
        # self.conn.sendall(data)
        self.msg = data  # data.decode("utf-8")
        #print(self.msg)
        self.set()

    def isFlag(self):
        return self.flag

    def set(self):
        self.flag = True

    def reset(self):
        self.flag = False

    def get_msg(self):
        return self.msg


sck = SockFlag()


def thread_sock():
    sck.connect()
    while True:
        if not sck.flag:
            data = sck.loop()
            print('Received Bytearray: len %d' % len(data))
            if data:
                sck.Process(data)
            if not data:
                sck.flag = False
                sck.connect()
                # pass #break

            # sck.Process(data)


def main():
    threadA = Thread(target=thread_sock)
    threadA.start()
    gps_data = None
    print("1")
    while True:
      if sck.isFlag():
        #gps_data = (51.758564, -1.256382)

        msg = sck.get_msg()
        lat = struct.unpack('d', msg[0:8])
        lon = struct.unpack('d', msg[8:])
        gps_data = (lat[0], lon[0])
        print("data = ", gps_data)
        sck.reset()

        if gps_data:
            vis = GPSVis(data_path=gps_data)

            vis.create_image(width=4)
            vis.plot_map()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


