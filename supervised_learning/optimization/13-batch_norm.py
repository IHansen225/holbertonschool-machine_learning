#!/usr/bin/env python3
"""
    Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Normalizes an unactivated
        output of a neural network
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Znorm = (Z - mean) / np.sqrt(var + epsilon)
    Znorm = gamma * Znorm + beta
    return Znorm
