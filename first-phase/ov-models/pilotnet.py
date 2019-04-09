import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D
import numpy as np

np.random.seed(1000)
# Instantiate an empty model
global model

model = Sequential()

model.add(Conv2D(filters=24, kernel_size=(5, 5), padding='valid', activation='relu', strides=(2, 2), input_shape=(227, 227, 3)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='valid', activation='relu', strides=(2, 2)))
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu', strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=(1, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=(1, 1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('linear'))

# model.summary()

# Compile the model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['mse', 'mae'])
