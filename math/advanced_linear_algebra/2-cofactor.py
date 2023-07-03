#!/usr/bin/env python3
"""
    matrix cofactor module
"""


def cofactor(matrix):
    """
        Returns the matrix cofactor
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return [[1]]
    cofactor_matrix = [[0] * len(matrix) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            minor = [[matrix[row][col] for col in range(len(matrix)) if col != j] for row in range(len(matrix)) if row != i]
            cofactor_matrix[i][j] = (-1) ** (i + j) * determinant(minor)
    return cofactor_matrix

def determinant(matrix):
    """
        Returns the determinant of a matrix
    """
    if len(matrix) == 1:
        return matrix[0][0]
    det = 0
    for j in range(len(matrix)):
        minor = [[matrix[i][col] for col in range(len(matrix)) if col != j] for i in range(1, len(matrix))]
        det += (-1) ** j * matrix[0][j] * determinant(minor)
    return det
