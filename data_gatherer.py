#import numpy as np
import cv2
import time
import tables
import sys

IMAGE_SIZE = (600, 400)
SLEEP_TIME = 0.05  # 20 frames/second

if len(sys.argv) < 2:
    sys.exit('No filename argument given')
file_name = sys.argv[1]
if '.' in file_name:
    index = file_name.index('.')
    file_name = file_name[:index]
PREFIX = '/media/ubuntu/3e8f0351-b1d8-4997-a57f-dca9d2e39b6c/home/mihai/6feb/'
FILENAME = PREFIX + file_name + str(int(time.time()) % 1000) + '.h5'

port = sys.argv[2]


def data_gatherer(ser):
    if ser is None:
        return False

    f = tables.open_file(FILENAME, mode='w')
    image_atom = tables.UInt8Atom()
    other_atom = tables.UInt16Atom()

    image_data = f.create_earray(f.root, 'image', image_atom, (0, IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    servo_data = f.create_earray(f.root, 'servo', other_atom, (0, 1))
    esc_data = f.create_earray(f.root, 'esc', other_atom, (0, 1))

    f.close()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            start = time.time()
            ret, img = cap.read()

            # cv2.imshow("input", img)
            # cv2.waitKey(10)

            # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)

            print("image size: " + str(img.shape))

            ser.write(b"0")
            #while ser.inWaiting() == 0:
            #    continue

            servo = ser.readline()
            if servo != '':
                servo = int(bytes.decode(servo))
            else:
                servo = 0
            
            esc = ser.readline()
            if esc != '':
                esc = int(bytes.decode(esc))
            else:
                esc = 0

            print("servo: "+str(servo) + "  esc: "+str(esc))

            f = tables.open_file(FILENAME, mode='a')

            f.root.image.append([img])
            f.root.servo.append([[servo]])
            f.root.esc.append([[esc]])

            f.close()

            print(SLEEP_TIME-(time.time()-start))

            if SLEEP_TIME-(time.time()-start) > 0:
                time.sleep(SLEEP_TIME-(time.time()-start))

    except Exception as e:
        print("ERROR: "+str(e))
#        f.flush()
        f.close()


if __name__ == "__main__":
    import serial
    ser = serial.Serial(port, baudrate=9600, timeout=0.01)
    while ser.inWaiting() == 0:
        continue
    msg = ser.readline()
    msg = bytes.decode(msg)
    if "connected" not in msg:
        sys.exit('Oups' + msg)
    data_gatherer(ser)
