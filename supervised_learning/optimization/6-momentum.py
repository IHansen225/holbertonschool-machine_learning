#!/usr/bin/env python3
"""
    Momentum optimization module.
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
        Updates a variable using
        the gradient descent with
        momentum optimization algorithm.
    """
    m_op = tf.train.MomentumOptimizer(alpha, beta1)
    return m_op.minimize(loss)
