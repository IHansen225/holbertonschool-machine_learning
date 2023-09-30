#!/usr/bin/env python3
"""
    Module to create a confusion
    matrix based on the provided
    data.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Returns a confusion matrix.
    """
    return np.matmul(labels.T, logits)
