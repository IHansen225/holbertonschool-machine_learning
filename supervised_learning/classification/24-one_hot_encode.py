#!/usr/bin/env python3
"""
    One hot encode module
"""
import numpy as np


def one_hot_encode(Y, cl):
    """
        Returns a one-hot encoded version
        of a numeric label vector.
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(cl, int) or cl <= np.amax(Y):
        return None
    res = [[1. if i == j else 0. for j in range(cl)] for i in Y]
    return np.array(res).T
