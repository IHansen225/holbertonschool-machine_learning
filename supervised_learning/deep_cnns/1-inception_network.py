#!/usr/bin/env python3
"""
    Inception DCNN network.
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
        Builds the inception network.
    """
    act = 'relu'
    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', activation=act,
                            kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(conv1)
    conv2R = K.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                             padding='same', activation=act,
                             kernel_initializer=init)(pool1)
    conv2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation=act,
                            kernel_initializer=init)(conv2R)
    pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(conv2)
    incept3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    incept3b = inception_block(incept3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(incept3b)
    incept4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    incept4b = inception_block(incept4a, [160, 112, 224, 24, 64, 64])
    incept4c = inception_block(incept4b, [128, 128, 256, 24, 64, 64])
    incept4d = inception_block(incept4c, [112, 144, 288, 32, 64, 64])
    incept4e = inception_block(incept4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(incept4e)
    incept5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    incept5b = inception_block(incept5a, [384, 192, 384, 48, 128, 128])
    avgPool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                        padding='valid')(incept5b)
    dropOut = K.layers.Dropout(0.4)(avgPool)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=init)(dropOut)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
