#!/usr/bin/env python3
"""
    L2 new layer.
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a L2 regularization layer.
    """
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    return layer(prev)
