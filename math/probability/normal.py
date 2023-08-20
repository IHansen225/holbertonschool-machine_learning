#!/usr/bin/env python3
"""
    Normal distribution class
"""


class Normal():
    """
        Normal distribution object
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = stddev
            self.mean = mean
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(self.norm_mean(data))
            self.stddev = float(self.norm_stddev(data))

    def error_f(self, x):
        """
            Computes the error value
        """
        pi = 3.1415926536
        x3 = (x ** 3) / 3
        x5 = (x ** 5) / 10
        x7 = (x ** 7) / 42
        x9 = (x ** 9) / 216
        return ((2/(pi**(1/2))) * (x - x3 + x5 - x7 + x9))

    def norm_mean(self, data):
        """
            Returns the mean of the given object's data
        """
        return sum(data)/len(data)

    def norm_stddev(self, data):
        """
            Returns the standard deviation of the given
            object's data
        """
        div = 0
        n = len(data)
        m = self.norm_mean(data)
        for i in range(0, len(data)):
            div += (data[i] - m)**2
        return (div / n)**(1/2)

    def z_score(self, x):
        """
            Returns the z-score of a given x point
        """
        return (x - self.norm_mean(self.data))/self.norm_stddev(self.data)

    def x_value(self, z):
        """
            Returns the x-score of a given z point
        """
        return z * self.norm_stddev(self.data) + self.norm_mean(self.data)

    def pdf(self, x):
        """
            Returns the probability density of a given
            x point in a normal distribution
        """
        pi = 3.1415926536
        e = 2.7182818285
        div = 1 / (self.norm_stddev(self.data) * ((2 * pi)**(1/2)))
        mean = self.norm_mean(self.data)
        exp = (-1 * 1/2) * ((x - mean) / self.norm_stddev(self.data)) ** 2
        return div * e ** exp

    def cdf(self, x):
        """
            Returns the cumulative probability of the
            given x point in a normal distribution
        """
        mean = self.norm_mean(self.data)
        stddev = self.norm_stddev(self.data)
        return 0.5 * (1 + self.error_f((x - mean)/(stddev * (2 ** 0.5))))
