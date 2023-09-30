#!/usr/bin/env python3
"""
    RMSProp optimization algorithm
    module.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        Updates the given variables with
        the RMSProp optimization algorithm.
    """
    sdw = beta2 * s + (1 - beta2) * (grad ** 2)
    W = var - alpha * (grad / (np.sqrt(sdw) + epsilon))
    return W, sdw
