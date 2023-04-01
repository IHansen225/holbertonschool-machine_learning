#!/usr/bin/env python3
"""
    Summation function
"""


def summation_i_squared(n):
    """
        Returns summation of given number
        as i^2
    """
    if n < 1:
        return None
    return (n * (n + 1) * ((2 * n) + 1)) / 6
