#!/usr/bin/env python3
"""
    Function to create a layer
    for a neural network in tensorflow.
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
        Returns the tensor output of the layer.
    """
    # Define the initializer to the He et al. method.
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    # Define the layer.
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name="layer")
    # Return the layer giving it the previous layer as input.
    return layer(prev)
