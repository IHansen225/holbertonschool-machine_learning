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
    try:
        ret = np.apply_along_axis(lambda row: np.argmax(row), axis=1, arr=oh)
    except Exception:
        return None
