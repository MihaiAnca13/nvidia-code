from keras import layers, models, optimizers
from keras.layers import Activation, Dense, Flatten
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import numpy as np

global model

x = layers.Input(shape=(227, 227, 3))

# Layer 1: Just a conventional Conv2D layer
conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

# Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

# Layer 3: Capsule layer. Routing algorithm works here.
digitcaps = CapsuleLayer(num_capsule=2, dim_capsule=16, routings=3,
                         name='digitcaps')(primarycaps)

flat = Flatten()(digitcaps)
output = Dense(2)(flat)
outa = Activation('linear')(output)

# Models for training and evaluation (prediction)
model = models.Model(x, outa, name='capsnet-v3')

# Compile the model
model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001), metrics=['mse', 'mae'])