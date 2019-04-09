from keras.layers import Reshape, Dense, Merge
from keras import backend as K

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend as K
from ..engine.base_layer import Layer
from ..engine.base_layer import InputSpec
from ..legacy import interfaces


class _GlobalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    @interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(_GlobalPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling2D(_GlobalPooling2D):
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.mean(inputs, axis=[1, 2])
        else:
            return K.mean(inputs, axis=[2, 3])


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', init='he_normal', bias=False)(se)
    se = Dense(filters, activation='sigmoid', init='he_normal', bias=False)(se)

    x = Merge([init, se], mode='mul')
    return x
