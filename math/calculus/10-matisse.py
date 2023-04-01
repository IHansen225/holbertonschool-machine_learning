#!/usr/bin/env python3
"""
    Derivative function
"""


def poly_derivative(poly):
    """
        Returns the derivative of the given poly
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [0] * (len(poly) - 1)
    for i in range(1, len(poly)):
        power = i
        coeff = poly[i]
        derivative[power - 1] = power * coeff
    while derivative and derivative[-1] == 0:
        derivative.pop()
    return derivative
