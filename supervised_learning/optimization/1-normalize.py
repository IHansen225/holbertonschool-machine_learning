#!/usr/bin/env python3
"""
    Module to normalize a
    given matrix.
"""
import numpy as np


def normalize(X, m, s):
    """
        Normalizes a matrix with
        a given stddev and mean.
    """
    return (X - m) / s
