#!/usr/bin/env python3
"""
    Matrix definiteness module.
"""
import numpy as np


def definiteness(matrix):
    """
        Calculates the definiteness of a matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues, _ = np.linalg.eig(matrix)
    num_positive = np.sum(eigenvalues > 0)
    num_zero = np.sum(np.isclose(eigenvalues, 0))

    if num_positive == matrix.shape[0]:
        return "Positive definite"
    elif num_positive == 0 and num_zero > 0:
        return "Negative definite"
    elif num_positive == matrix.shape[0] - num_zero:
        return "Positive semi-definite"
    elif num_positive == 0 and num_zero < matrix.shape[0]:
        return "Negative semi-definite"
    else:
        return "Indefinite"
