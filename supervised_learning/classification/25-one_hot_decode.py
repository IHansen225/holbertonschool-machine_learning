#!/usr/bin/env python3
"""
    One hot decode module
"""
import numpy as np


def one_hot_decode(oh):
    """
        Returns a one-hot decoded version
        of a numeric label vector.
    """
    if not isinstance(oh, np.ndarray):
        return None
    if oh.size == 0:
        return None
    if not np.all(np.sum(oh, axis=1) == 1):
        return None
    return np.argmax(oh, axis=1)
