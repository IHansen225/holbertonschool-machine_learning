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
    