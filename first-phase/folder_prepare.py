import numpy as np
import tables
import cv2
import os
import glob

np.set_printoptions(threshold=np.nan)

foldername = 'D:\\rc car data collection\\second\\'
result = "D:\\rc car data collection\\second\\alldata\\alldata.h5"

if os.path.isfile(result):
    os.remove(result)

g = tables.open_file(result, mode='a')

image_atom = tables.UInt8Atom()
other_atom = tables.Float32Atom()

image_data = g.create_earray(g.root, 'image', image_atom, (0, 227, 227, 3))
servo_data = g.create_earray(g.root, 'servo', other_atom, (0, 1))
esc_data = g.create_earray(g.root, 'esc', other_atom, (0, 1))

for item in glob.glob(os.path.join(foldername, '*.h5')):
    print('Doing '+str(item))

    f = tables.open_file(item, mode='r')

    images = np.array(f.root.image)
    servo = np.array(f.root.servo, dtype=np.float)
    esc = np.array(f.root.esc, dtype=np.float)

    f.close()

    image_values = []

    for i in range(len(images)):
        img = cv2.resize(images[i], dsize=(227, 227), interpolation=cv2.INTER_AREA)
        image_values.append(img)

    g.root.image.append(image_values)
    g.root.servo.append(servo)
    g.root.esc.append(esc)

g.close()