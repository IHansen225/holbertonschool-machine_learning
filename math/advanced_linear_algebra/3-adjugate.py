#!/usr/bin/env python3
"""
    matrix adjugate module
"""


def adjugate(matrix):
    """
    Returns the matrix adjugate
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    cofactor = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [row[:j] + row[j+1:]
                     for row in (matrix[:i] + matrix[i+1:])]
            cofactor[i][j] = (-1) ** (i+j) * determinant(minor)
    return transpose(cofactor)


def determinant(matrix):
    """
    Returns the determinant of a matrix
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    sign = 1
    for i in range(n):
        submatrix = [row[1:] for row in matrix[1:]]
        det += sign * matrix[0][i] * determinant(submatrix)
        sign *= -1
    return det


def transpose(matrix):
    """
    Returns the transpose of a matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
