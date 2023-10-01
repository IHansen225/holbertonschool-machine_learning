#!/usr/bin/env python3
"""
    Creates a Dropout layer.
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a Dropout layer.
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dropout(rate=keep_prob)
    return layer(prev)
