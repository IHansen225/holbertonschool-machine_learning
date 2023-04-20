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
    if not isinstance(oh, np.ndarray) or len(oh) == 0:
        return None
    if not np.any(oh == 1, axis=0).all():
        return None
    return np.argmax(oh, axis=0)
