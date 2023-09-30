#!/usr/bin/env python3
"""
    Shuffles arrays.
"""
import numpy as np


def shuffle_data(X, Y):
    """
        Returns the shuffled forms
        of the given arrays.
    """
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]
