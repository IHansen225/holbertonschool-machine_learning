#!/usr/bin/env python3
"""
    Updates a variable using the Adam
    optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        Updates a variable using the Adam
        optimization algorithm.
    """
    vdw = beta1 * v + (1 - beta1) * grad
    vdw_corr = vdw / (1 - beta1 ** t)
    sdw = beta2 * s + (1 - beta2) * (grad ** 2)
    sdw_corr = sdw / (1 - beta2 ** t)
    W = var - alpha * (vdw_corr / (np.sqrt(sdw_corr) + epsilon))
    return W, vdw, sdw
