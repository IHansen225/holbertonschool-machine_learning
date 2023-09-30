#!/usr/bin/env python3
"""
    Module to calculate the class
    sensitivity for each confusion
    matrix element.
"""
import numpy as np


def sensitivity(confusion):
    """
        Returns an array containing
        the sensitivity of each class.
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
