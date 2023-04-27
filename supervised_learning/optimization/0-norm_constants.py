#!/usr/bin/env python3
"""
    Module to calculate the
    normalization constants of
    a given matrix.
"""
import numpy as np


def normalization_constants(X):
    """
        Calculates the normalization
        constants of a matrix.
        Returns the mean and standard
        deviation of each feature.
    """
    return X.T.mean(axis=1), X.T.std(axis=1)
