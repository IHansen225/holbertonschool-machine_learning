#!/usr/bin/env python3
"""
    The add_matrices2D(mat1, mat2) function returns
    the added version of the two given matrices.
"""


def add_matrices2D(mat1, mat2):
    """
        Adds the two given matrices.
        If the two matrices are not the same
        shape, does nothing.
    """
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    nmat = [[] for i in range(len(mat1))]
    for i in range(len(nmat)):
        for j in range(len(mat1[i])):
            nmat[i].append(mat1[i][j] + mat2[i][j])
    return nmat
