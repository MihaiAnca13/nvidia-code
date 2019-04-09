import tables
import cv2
import numpy as np
import time
from keras.models import model_from_json
import csv

name = "models/seresnet-first-train"

# load json and create model
json_file = open(name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(name+".h5")
print("Loaded model from disk")

# load data
train_file = "D:\\rc car data collection\\second\\d2-141.h5"

f = tables.open_file(train_file, mode='r')

images = np.array(f.root.image)
servo = np.array(f.root.servo, dtype=np.float)
esc = np.array(f.root.esc, dtype=np.float)

with open(name+'-results3.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Target', 'Prediction'])
    for i in range(len(images)):
        img = cv2.resize(images[i], dsize=(227, 227), interpolation=cv2.INTER_AREA)
        prediction = model.predict(np.array([img]), batch_size=1)
        target_servo = servo[i][0]
        predicted_servo = prediction[0][0]
        csv_writer.writerow([target_servo, predicted_servo])
