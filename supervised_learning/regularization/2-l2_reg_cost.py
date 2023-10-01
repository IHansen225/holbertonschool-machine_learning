#!/usr/bin/env python3
"""
    L2 Regularization gradient
    descent module.
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
        Returns: a tensor containing the L2 regularization cost.
    """
    return cost + tf.losses.get_regularization_losses()
