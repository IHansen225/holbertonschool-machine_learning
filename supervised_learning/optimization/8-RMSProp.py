#!/usr/bin/env python3
"""
    RMSProp optimization algorithm
    module.
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the training operation for
        a neural network in tensorflow using
        the RMSProp optimization algorithm.
    """
    return tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    ).minimize(loss)
