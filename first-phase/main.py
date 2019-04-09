import numpy as np
import tables
from random import shuffle
import os

np.set_printoptions(threshold=np.nan)

filename = "D:\\rc car data collection\\second\\alldata\\alldata.h5"
train_data = "D:\\rc car data collection\\second\\alldata\\train_data.h5"

f = tables.open_file(filename, mode='r')

images = np.array(f.root.image)
servo = np.array(f.root.servo, dtype=np.float)
esc = np.array(f.root.esc, dtype=np.float)
# timestamps = np.array(f.root.timestamp)

f.close()

if os.path.isfile(train_data):
    os.remove(train_data)

g = tables.open_file(train_data, mode='w')

image_atom = tables.UInt8Atom()
other_atom = tables.Float32Atom()

image_data = g.create_earray(g.root, 'image', image_atom, (0, 227, 227, 3))
servo_data = g.create_earray(g.root, 'servo', other_atom, (0, 1))
esc_data = g.create_earray(g.root, 'esc', other_atom, (0, 1))

# define boundaries
max_speed = max(esc) + 1
min_speed = min(esc) - 1
max_right = min(servo) - 1
max_left = max(servo) + 1

servo_values = []
esc_values = []
# convert to 0-1 range and resize images
for i in range(len(servo)):
    v = ((servo[i] - max_right) / (max_left - max_right)).item()
    servo_values.append(v)
    v = ((esc[i] - min_speed) / (max_speed - min_speed)).item()
    esc_values.append(v)

del servo, esc

dataset = []
for i in range(images.shape[0]):
    dataset.append([images[i], servo_values[i], esc_values[i]])

shuffle(dataset)

for i in range(len(dataset)):
    g.root.image.append([dataset[i][0]])
    g.root.servo.append([[dataset[i][1]]])
    g.root.esc.append([[dataset[i][2]]])

# max speed: 1565
# min speed: 1459
# min right: 1063
# max left: 1949

g.close()
print('Done')