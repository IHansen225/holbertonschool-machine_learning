#!/usr/bin/env python3
"""
    Posterior probability
    module
"""


import numpy as np


def posterior(x, n, P, Pr):
    """
        Retuns the posterior probability
        for the various hypothetical
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
            )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not all(0 <= x <= 1 for x in P):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if not all(0 <= x <= 1 for x in Pr):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    likelihood = Pr * (
        np.math.factorial(n) /
        (np.math.factorial(x) * np.math.factorial(n - x))
        ) * P ** x * (1 - P) ** (n - x)
    evidence = np.sum(likelihood)
    return likelihood / evidence