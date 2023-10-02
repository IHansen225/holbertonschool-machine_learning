#!/usr/bin/env python3
"""
    Identity DCNN block.
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
        Defines an identity block.
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(F11, (1, 1), padding='same',
                            kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=init)(act1)
    norm2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(norm2)
    conv3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                            kernel_initializer=init)(act2)
    norm3 = K.layers.BatchNormalization()(conv3)
    add = K.layers.Add()([norm3, A_prev])
    output = K.layers.Activation('relu')(add)
    return output
