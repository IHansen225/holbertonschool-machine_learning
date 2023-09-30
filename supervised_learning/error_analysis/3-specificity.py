#!/usr/bin/env python3
"""
    Module to calculate the class
    specificity for each cnf
    matrix element.
"""
import numpy as np


def specificity(cnf):
    """
        Returns an array containing
        the specificity of each class.
    """
    true_negatives = np.sum(cnf) - np.sum(cnf, axis=0) -\
        np.sum(cnf, axis=1) + np.diagonal(cnf)
    return true_negatives / (np.sum(cnf) - np.sum(cnf, axis=1))
