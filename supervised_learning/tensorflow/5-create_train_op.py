#!/usr/bin/env python3
"""
    Train module.
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
        Creates the training operation for the network.
    """
    train = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train
