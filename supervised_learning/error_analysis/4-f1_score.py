#!/usr/bin/env python3
"""
    Module to calculate the class
    f1 score for each cnf
    matrix element.
"""
import numpy as np


def f1_score(cnf):
    """
        Returns an array containing
        the f1 score of each class.
    """
    return 2 * np.diagonal(cnf) / (np.sum(cnf, axis=0) +
                                   np.sum(cnf, axis=1))
