#!/usr/bin/env python3
"""
    Batch Normalization
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
        Normalizes an unactivated
        output of a neural network
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=init,
        name="layer"
    )
    Z = layer(prev)
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]),
        name="gamma"
    )
    beta = tf.Variable(
        tf.constant(0.0, shape=[n]),
        name="beta"
    )
    mean, var = tf.nn.moments(Z, axes=[0])
    epsilon = tf.constant(1e-8)
    Znorm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=var,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
    return activation(Znorm)
