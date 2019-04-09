#!/usr/bin/env python

import sys
from socket import *
from datetime import datetime
import serial
import time

read_mode = False
host = "10.42.0.209"
port = 13000
buf = 1024
addr = (host, port)
UDPSock = socket(AF_INET, SOCK_DGRAM)

while True:
    try:
        UDPSock.bind(addr)
        ser = serial.Serial('/dev/ttyUSB0', baudrate=9600)
        print("Start")
        while True:
            (data, addr) = UDPSock.recvfrom(buf)
            data = bytes.decode(data)
            with open('/home/ubuntu/nvidia-code/command-log.log', 'a') as f:
                f.write(str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")) + ' - ' + data + '\n')
            if data == "exit":
                UDPSock.close()
                sys.exit(0)
            elif data == "start":
                ser.write('readings?')
                while ser.inWaiting() == 0:
                    continue
                response = ser.readline()
                UDPSock.sendto(response, addr)

    except Exception as e:
        with open('/home/ubuntu/nvidia-code/error-log.log', 'a') as f:
            f.write(str(e)+'\n')
            # wait 15 seconds
            time.sleep(15)
