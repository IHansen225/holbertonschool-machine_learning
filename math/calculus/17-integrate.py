#!/usr/bin/env python3
"""
    Integration function
"""


def poly_integral(poly, C=0):
    """
        Returns the integral of the given poly
    """
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    for i in poly:
        if type(i) is not int and type(i) is not float:
            return None
    integral = [C]
    for i in range(len(poly)):
        if poly[i] % (i + 1) == 0:
            integral.append(poly[i] // (i + 1))
        else:
            integral.append(poly[i] / (i + 1))
    while integral[-1] == 0 and len(integral) > 1:
        integral.pop()
    return integral
