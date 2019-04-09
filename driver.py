import cv2
from time import sleep, time
from keras.models import model_from_json
import numpy as np
import serial
import sys

SLEEP_TIME = 0.05  # 20 frames/second
SLEEP_TIME = SLEEP_TIME * 4  # every 4th step => 5 frames/second

def drive(model, ser, cap):
    while cap.isOpened():
        start = time()
        ret, img = cap.read()
        img = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_AREA)

        prediction = model.predict(np.array([img]), batch_size=1)
        print(prediction)
        servo, esc = prediction[0]

        servo = int(servo)
        esc = int(esc)

        ser.write(b'0')
        ser.write((str(servo) + 'S' + str(esc) + 'E').encode())

        st = time() - start
        st = SLEEP_TIME - st
        print('zzz for {} s'.format(st))
        if st < 0:
            st = 0
        sleep(st)


if len(sys.argv) < 2:
    sys.exit('no filename')
model_name = sys.argv[1]

if len(sys.argv) < 3:
    sys.exit('no port')
port = sys.argv[2]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
if img is None:
    print('img is none')
    sys.exit(0)

json_file = open(model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_name + ".h5")
print("Loaded model from disk")

ser = serial.Serial(port, baudrate=115200, timeout=0.01)
while ser.inWaiting() == 0:
    continue
msg = ser.readline()
msg = bytes.decode(msg)

if "connected" not in msg:
    sys.exit("oups" + str(msg))

drive(model, ser, cap)
