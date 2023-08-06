#!/usr/bin/env python3
"""
    The matrix_shape(matrix) function returns
    a list containing the dimensions of the
    recieved matrix.
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
