#!/usr/bin/env python3
"""
    Densenet 121 structure module.
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth, rate=32, compression=1.0):
    """
        Creates a DenseNet-121 architecture.
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=2*rate,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding='same',
                           kernel_initializer=init)(activation)
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same')(conv)
    X, nb_filters = dense_block(max_pool, 2*rate, growth, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth, 16)
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding='valid')(X)
    dense = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init)(avg_pool)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
