#!/usr/bin/env python3
"""
    Derivative function
"""


def poly_derivative(poly):
    """
        Returns the derivative of the given poly
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    for i in poly:
        if type(i) is not int and type(i) is not float:
            return None
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    if derivative == []:
        derivative.append(0)
    return derivative
