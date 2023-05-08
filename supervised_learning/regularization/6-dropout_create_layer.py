#!/usr/bin/env python3
"""
    Creates a Dropout layer.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a Dropout layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dropout(keep_prob)
    return layer(prev)
