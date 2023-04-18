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
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        if not len(list(filter(lambda x: x > 0, layers))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        if not len(list(filter(lambda x: isinstance(x, int), layers))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            bN = "b{}".format(l + 1)
            self.weights[bN] = np.zeros((layers[l], 1))
            nnx = nx if l == 0 else layers[l - 1]
            wN = "W{}".format(l + 1)
            self.weights[wN] = np.random.randn(layers[l], nnx) * np.sqrt(2 / nnx)
