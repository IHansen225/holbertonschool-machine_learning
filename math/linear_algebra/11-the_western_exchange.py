#!/usr/bin/env python3
"""
    Numpy function to transpose the
    given matrix.
"""


def np_transpose(matrix):
    matrixT = [[] for i in range(len(matrix[0]))]
    for i in range(len(matrixT)):
        for j in range(len(matrix)):
            matrixT[i].append(matrix[j][i])
    return matrixT
