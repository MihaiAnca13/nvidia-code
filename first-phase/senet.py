from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import multiply
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import Adam
from keras import backend as K

from se import squeeze_excite_block

__all__ = ['SEResNet', 'SEResNet50', 'SEResNet101', 'SEResNet154', 'preprocess_input', 'decode_predictions']

WEIGHTS_PATH = ""
WEIGHTS_PATH_NO_TOP = ""

global model

depth = [3, 4, 6, 3]
#filters = [64, 128, 256, 512]
filters = [32, 64, 128, 256]
initial_conv_filters = 64
channel_axis = -1
weight_decay = 1e-4
width = 1
outputs = 2


def _resnet_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block without bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


img_input = Input(shape=(227, 227, 3))
# block 1 (initial conv block)
x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
           kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

N = list(depth)
# block 2 (projection block)
for i in range(N[0]):
    x = _resnet_block(x, filters[0], width)

# block 3 - N
for k in range(1, len(N)):
    x = _resnet_block(x, filters[k], width, strides=(2, 2))

    for i in range(N[k] - 1):
        x = _resnet_block(x, filters[k], width)

x = BatchNormalization(axis=channel_axis)(x)
x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(outputs, use_bias=False, kernel_regularizer=l2(weight_decay),
          activation='linear')(x)

model = Model(img_input, x, name='se-res-net-50')
model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mse', 'mae'])
model.summary()