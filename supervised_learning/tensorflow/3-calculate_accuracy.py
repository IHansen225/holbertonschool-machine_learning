#!/usr/bin/env python3
"""
    Accuracy calculation module.
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction.
    """
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    return accuracy
