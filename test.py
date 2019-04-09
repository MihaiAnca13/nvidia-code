import tables
import cv2
import numpy as np
import time

filename = "D:\\rc car data collection\\second\\alldata\\train_data.h5"
# filename = 'test01436.h5'

# ROW_SIZE = 3
# NUM_COLUMNS = 200

# f = tables.open_file(filename, mode='w')
# atom = tables.Float64Atom()
#
# array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))
#
# for idx in range(NUM_COLUMNS):
#     x = np.random.rand(1, ROW_SIZE)
#     print(x.shape)
#     array_c.append(x)
# f.close()

# f = tables.open_file(filename, mode='a')
# f.root.data.append(x)
# f.close()

f = tables.open_file(filename, mode='r')

images = np.array(f.root.image)
servo = np.array(f.root.servo)
f.close()

print(len(images))

nr = 0
for i in range(len(images)):
    image = images[i]
    srv = servo[i]
    if srv == 0:
        continue
    print(i)
    if 1450 <= srv <= 1570:
        image = cv2.arrowedLine(image, (25, 50), (25, 10), (255, 255, 0))
        print('fwd\n')
    elif srv < 1300:
        image = cv2.arrowedLine(image, (10, 10), (50, 10), (255,255,0))
        print('right\n')
    elif srv > 1700:
        image = cv2.arrowedLine(image, (50, 10), (10, 10), (255, 255, 0))
        print('left\n')
    nr += 1
    # print(nr)
    cv2.imshow('img', image)
    time.sleep(0.5)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        stop = input()
        if stop == 'y':
            exit()
cv2.destroyAllWindows()