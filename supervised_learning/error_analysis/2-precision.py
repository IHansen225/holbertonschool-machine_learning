#!/usr/bin/env python3
"""
    Module to calculate the class
    precision for each confusion
    matrix element.
"""
import numpy as np


def precision(confusion):
    """
        Returns an array containing
        the precision of each class.
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
