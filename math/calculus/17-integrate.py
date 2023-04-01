#!/usr/bin/env python3
"""
    Integration function
"""


def poly_integral(poly, C=0):
    """
        Returns the integral of the given poly
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        if coeff.is_integer():
            coeff = int(coeff)
        integral.append(coeff)

    # Remove trailing zeros (if any)
    while integral and integral[-1] == 0:
        integral.pop()

    return integral
