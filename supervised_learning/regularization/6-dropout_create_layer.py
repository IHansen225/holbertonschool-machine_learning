#!/usr/bin/env python3
"""
    Creates a Dropout layer.
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        Creates a Dropout layer.
    """
    w = tf.Variable(tf.random.normal([prev.shape[1], n]))
    b = tf.Variable(tf.zeros([n]))
    prev_drop = tf.nn.dropout(prev, rate=1-keep_prob)
    layer = activation(tf.matmul(prev_drop, w) + b)
    return layer
