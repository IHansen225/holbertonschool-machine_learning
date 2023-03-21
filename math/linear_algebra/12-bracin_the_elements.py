#!/usr/bin/env python3
"""
    Numpy function to return the
    element-wise sum, sub, mul and div
    of the given matrices.
"""


def np_elementwise(mat1, mat2):
    """
        Performs element-wise addition, 
        subtraction, multiplication,
        and division.
    """
    return (np.add(mat1, mat2),
            np.subtract(mat1, mat2),
            np.multiply(mat1, mat2),
            np.divide(mat1, mat2))
