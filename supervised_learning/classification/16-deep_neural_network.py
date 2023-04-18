#!/usr/bin/env python3
"""
    Deep neural network class
    file for supervised learning
    implementation.
"""
import numpy as np


class DeepNeuralNetwork():
    """
        Deep neural network class.
    """
    def __init__(self, nx, ly):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(ly, list) or not ly:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(ly)
        if not len(list(filter(lambda x: x > 0, ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        if not len(list(filter(lambda x: isinstance(x, int), ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        self.cache = {}
        self.weights = {}
        for lyr in range(self.L):
            bN = "b{}".format(lyr + 1)
            self.weights[bN] = np.zeros((ly[lyr], 1))
            nnx = nx if lyr == 0 else ly[lyr - 1]
            wN = "W{}".format(lyr + 1)
            self.weights[wN] = np.random.randn(ly[lyr], nnx) * np.sqrt(2 / nnx)
