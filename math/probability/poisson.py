#!/usr/bin/env python3
"""
    Poisson distribution class
"""


class Poisson(object):
    """
        Poisson distribution object
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
            Calculates probability mass of poisson distribution
        """
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        fk = 1
        for i in range(1, k+1):
            fk *= i
        e = 2.7182818285
        return ((e**(-1 * self.lambtha))*(self.lambtha**k))/(fk)

    def cdf(self, k):
        """
            Calculates the cumulative distribution of poisson distribution
        """
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        cdf = 0
        for i in range(0, k+1):
            cdf += self.pmf(i)
        return cdf
