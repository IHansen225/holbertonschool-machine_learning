#!/usr/bin/env python3
"""
    Module to calculate
    the definiteness of a matrix
"""


def determinant(matrix):
    """
        Returns the determinant of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(row):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [[r[n] for n in range(len(matrix)) if n != i] for r in rows]
        det += k * (-1)**i * determinant(new_m)
    return det


def definiteness(matrix):
    """
        Calculates the definiteness of a matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return "Positive definite"
    if len(matrix) == 2:
        a, b = matrix[0][0], matrix[0][1]
        c, d = matrix[1][0], matrix[1][1]
        if a > 0 and d > 0 and b * c > 0:
            return "Positive definite"
        if a < 0 and d < 0 and b * c > 0:
            return "Positive definite"
        if a < 0 and d < 0 and b * c < 0:
            return "Negative definite"
        if a > 0 and d > 0 and b * c < 0:
            return "Negative definite"
        if a == 0 or d == 0:
            return "Indefinite"
        return "Positive semi-definite"
    if len(matrix) == 3:
        a, b, c = matrix[0][0], matrix[0][1], matrix[0][2]
        d, e, f = matrix[1][0], matrix[1][1], matrix[1][2]
        g, h, i = matrix[2][0], matrix[2][1], matrix[2][2]
        if a > 0 and d > 0 and g > 0 and determinant(matrix) > 0:
            return "Positive definite"
        if a < 0 and d < 0 and g < 0 and determinant(matrix) > 0:
            return "Positive definite"
        if a < 0 and d < 0 and g < 0 and determinant(matrix) < 0:
            return "Negative definite"
        if a > 0 and d > 0 and g > 0 and determinant(matrix) < 0:
            return "Negative definite"
        if a == 0 or d == 0 or g == 0:
            return "Indefinite"
        return "Positive semi-definite"
