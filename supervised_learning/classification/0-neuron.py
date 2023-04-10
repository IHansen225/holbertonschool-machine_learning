#!/usr/bin/env python3
"""
    Neuron class file for classification
    algorithm exercises
"""


class Neuron():
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal()
        self.b = 0
        self.A = 0
