#!/usr/bin/env python3
"""
    The matrix_transpose(matrix) function returns
    the transposed version of the given matrix.
"""


def matrix_transpose(matrix):
    """
        Transposes the given matrix.
    """
    matrixT = [[] for i in range(len(matrix[0]))]
    for i in range(len(matrixT)):
        for j in range(len(matrix)):
            matrixT[i].append(matrix[j][i])
    return matrixT
