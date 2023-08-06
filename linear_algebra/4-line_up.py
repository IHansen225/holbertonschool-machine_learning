#!/usr/bin/env python3
"""
    The add_arrays(arr1, arr2) function returns
    the added version of the two given arrays.
"""


def add_arrays(arr1, arr2):
    """
        Adds the two given arrrays.
        If the two arrays are not the same
        shape, does nothing.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
