#!/usr/bin/env python3
"""
    Module to calculate the class
    specificity for each confusion
    matrix element.
"""
import numpy as np


def specificity(confusion):
    """
        Returns an array containing
        the specificity of each class.
    """
    true_negatives = np.sum(confusion) - np.sum(confusion, axis=0) -\
        np.sum(confusion, axis=1) + np.diagonal(confusion)
    return true_negatives / (np.sum(confusion) - np.sum(confusion, axis=1))
