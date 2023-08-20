#!/usr/bin/env python3
"""
    Exponential distribution class
"""


class Exponential():
    """
        Exponential distribution object
    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
            Probabilitiy density function for exponential distribution
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return (self.lambtha * (e ** (-1 * self.lambtha * x)))

    def cdf(self, x):
        """
            Cumulative density function for exponential distribution
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return (1 - (e ** (-1 * self.lambtha * x)))
