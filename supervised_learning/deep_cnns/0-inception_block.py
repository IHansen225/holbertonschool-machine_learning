#!/usr/bin/env python3
"""
    Inception DCNN block.
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
        Builds an inception block.
    """
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters
    ac = 'relu'
    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                            activation=ac, kernel_initializer=init)(A_prev)
    conv3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                             activation=ac, kernel_initializer=init)(A_prev)
    conv3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            activation=ac, kernel_initializer=init)(conv3R)
    conv5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                             activation=ac, kernel_initializer=init)(A_prev)
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                            activation=ac, kernel_initializer=init)(conv5R)
    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    convPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                             activation=ac, kernel_initializer=init)(pool)
    output = K.layers.concatenate([conv1, conv3, conv5, convPP])
    return output
