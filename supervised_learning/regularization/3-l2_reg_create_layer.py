#!/usr/bin/env python3
"""
    L2 new layer.
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Creates a L2 regularization layer.
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=w,
                            kernel_regularizer=l2)
    return layer(prev)
