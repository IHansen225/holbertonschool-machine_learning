#!/usr/bin/env python3
"""
    Creates a convolutional
    neural network using keras
    and the LeNet-5 architecture.
"""
import tensorflow.keras as K


def lenet5(X):
    """
        Builds a modified version
        of the LeNet-5 architecture.
    """
    init = K.initializers.he_normal(seed=None)
    activation = 'relu'
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation=activation, kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation=activation, kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)
    fc2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(fc1)
    fc3 = K.layers.Dense(units=10, kernel_initializer=init)(fc2)
    softmax = K.layers.Softmax()(fc3)
    model = K.models.Model(inputs=X, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
