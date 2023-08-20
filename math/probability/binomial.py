#!/usr/bin/env python3
"""
    Binomial distribution class
"""


class Binomial():
    """
        Binomial distribution object
    """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.p = float(p)
            self.n = int(n)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            m = sum(data) / len(data)
            res = 0
            for i in range(len(data)):
                res += (data[i] - m) ** 2
            var = res / len(data)
            self.n = round(m / (-(var/m) + 1))
            self.p = m / self.n

    def factorial(self, n):
        """
            Returns the factorial of a given integer n.
        """
        if n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""

        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        n = self.n
        p = self.p
        nk = (self.factorial(n)) / (self.factorial(k) * self.factorial(n - k))
        return (nk * (p ** k) * ((1 - p) ** (n - k)))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""

        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
