#!/usr/bin/env python3
"""
    DCNN dense block.
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
        Creates a DCNN dense block.
    """
    init = K.initializers.he_normal()
    for i in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation('relu')(batch_norm)
        conv = K.layers.Conv2D(filters=4*growth_rate,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=init)(activation)
        batch_norm = K.layers.BatchNormalization()(conv)
        activation = K.layers.Activation('relu')(batch_norm)
        conv = K.layers.Conv2D(filters=growth_rate,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=init)(activation)
        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate
    return X, nb_filters
