import serial
from time import time


ser = serial.Serial('COM3', baudrate=9600, timeout=1)

while True:
    a = input()
    ser.write(a.encode())
    print(ser.readline())
