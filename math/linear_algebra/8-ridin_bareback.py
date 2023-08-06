#!/usr/bin/env python3
"""
    The mat_mul(mat1, mat2) function returns
    the multiplied version of the two given matrices.
"""


def mat_mul(mat1, mat2):
    """
        Multiplies the two given matrices.
        If the two matrices cannot be
        multiplicated, it does nothing.
    """
    if len(mat1[0]) != len(mat2):
        return None
    nmat = [[0] * len(mat2[0]) for i in range(len(mat1))]
    for i in range(len(nmat)):
        for j in range(len(mat2[0])):
            nmat[i][j] = 0
            for k in range(len(mat2)):
                nmat[i][j] += mat1[i][k] * mat2[k][j]
    return nmat
