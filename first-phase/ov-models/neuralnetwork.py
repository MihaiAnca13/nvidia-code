import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
import numpy as np

np.random.seed(1000)
# Instantiate an empty model
global model

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(nb_filter=96, input_shape=(227, 227, 3), nb_row=11, nb_col=11, subsample=(4, 4), border_mode='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(nb_filter=256, nb_row=11, nb_col=11, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(nb_filter=384, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(nb_filter=384, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(nb_filter=256, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation('linear'))

# model.summary()

# Compile the model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['mse', 'mae'])

NAME = 'ov-alexnet'
# serialize model to JSON
model_json = model.to_json()
with open(NAME+".json", "w") as json_file:
    json_file.write(model_json)
