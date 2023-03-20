#!/usr/bin/env python3
"""
    The cat_matrices2D(mat1, mat2) function returns
    the added version of the two given matrices.
"""

def matrix_shape(matrix):
    """
        Caculates the shape of the provided matrix
    """
    dim = [len(matrix)]
    obj = matrix
    while True:
        if isinstance(obj[0], list):
            dim.append(len(obj[0]))
            obj = obj[0]
        else:
            return dim

def cat_matrices2D(mat1, mat2, axis=0):
    """
        Adds the two given matrices.
        If the two matrices cannot be
        concatenated, it does nothing.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if axis == 0:
        return mat1 + mat2
    return [mat1[i] + mat2[i] for i in range(len(mat1))]
