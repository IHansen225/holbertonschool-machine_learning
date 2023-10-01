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
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=init,
    )(X)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(conv1)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=init,
    )(pool1)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(conv2)
    flat = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=init,
    )(flat)
    dense2 = K.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=init,
    )(dense1)
    dense3 = K.layers.Dense(
        units=10,
        kernel_initializer=init,
    )(dense2)
    y_pred = K.layers.Softmax()(dense3)
    model = K.models.Model(inputs=X, outputs=y_pred)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
