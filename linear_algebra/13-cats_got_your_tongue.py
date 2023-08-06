#!/usr/bin/env python3
"""
    Numpy function to return the
    concatenated versions of the
    given matrices.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
        Concatenate the given matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
