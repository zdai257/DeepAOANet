#!/usr/bin/env python3

import socket
import struct

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

data = (51.758564, -1.256382)
lat = bytearray(struct.pack("d", data[0]))
lon = bytearray(struct.pack("d", data[1]))
msg = lat + lon
print(["0x%02x" % b for b in msg])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(msg)
    data = s.recv(1024)

print('Received', repr(data))