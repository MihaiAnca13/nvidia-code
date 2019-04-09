from neuralnetwork import model as model1
from pilotnet import model as model2
from senet import model as model3
from capsnet import model as model4
import numpy as np
import tables
import keras
import time

TRAIN_PERCENTAGE = 80
VALIDATION_PERCENTAGE = 15
TEST_PERCENTAGE = 15
NAME = "capsnet-first-train"
model = model4

np.random.seed(int(time.time()))

train_file = "D:\\rc car data collection\\second\\alldata\\alldata.h5"

f = tables.open_file(train_file, mode='r')

images = np.array(f.root.image)
servo = np.array(f.root.servo, dtype=np.float)
esc = np.array(f.root.esc, dtype=np.float)

train_data = []
for i in range(images.shape[0]):
    train_data.append([images[i], servo[i].item(), esc[i].item()])

del images, servo, esc

np.random.shuffle(train_data)
train_data = np.array(train_data)

data_size = train_data.shape[0]
train_size = int(data_size*TRAIN_PERCENTAGE/100)
validation_size = int(data_size*VALIDATION_PERCENTAGE/100)
test_size = int(data_size*TEST_PERCENTAGE/100)

X_train = np.stack(train_data[:, 0][:train_size])
Y_train = np.stack(train_data[:, 1:3][:train_size])

X_val = np.stack(train_data[:, 0][train_size:train_size+validation_size])
Y_val = np.stack(train_data[:, 1:3][train_size:train_size+validation_size])

X_test = np.stack(train_data[:, 0][-test_size:])
Y_test = np.stack(train_data[:, 1:3][-test_size:])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph/{}'.format(NAME), histogram_freq=0, write_graph=True, write_images=True)

prediction = model.predict(X_test[1:3], batch_size=1)
print(prediction)

# batch size = 64 for alexnet
results = model.fit(X_train, Y_train, batch_size=64, epochs=250, callbacks=[tbCallBack], validation_data=(X_val, Y_val))

prediction = model.predict(X_test[1:3], batch_size=1)
print(prediction)

eval_results = model.evaluate(X_test, Y_test, batch_size=1)
print(eval_results)

# serialize model to JSON
model_json = model.to_json()
with open(NAME+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(NAME+".h5")
print("Saved model to disk")
