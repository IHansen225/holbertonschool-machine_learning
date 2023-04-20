#!/usr/bin/env python3
"""
    Function to create placeholders
    for a neural network in tensorflow.
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
        Returns two placeholders, x and y,
        for the neural network.
    """
    # Placeholders are basically variables that you can feed data to.
    # Kinda like empty layers.
    # The first argument is the data type.
    # The second argument is the shape of the data.
    # The third argument is the name of the placeholder.
    x = tf.placeholder("float32", [None, nx], name="x")
    y = tf.placeholder("float32", [None, classes], name="y")
    return x, y
