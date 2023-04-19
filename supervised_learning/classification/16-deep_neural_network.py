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
        # Number of layers in the neural network.
        self.L = len(ly)
        # Checks if all the elements in ly are positive integers.
        # Lambda function to avoid using loops.
        if not len(list(filter(lambda x: x > 0, ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        # Checks if all the elements in ly are integers.
        # Same thing as above.
        if not len(list(filter(lambda x: isinstance(x, int), ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        self.cache = {}
        self.weights = {}
        # Weight and bias initialization.
        # The weights are initialized using He et al. method.
        # The biases are initialized to 0.
        # The weights are initialized to a random normal distribution.
        for lyr in range(self.L):
            bN = "b{}".format(lyr + 1)
            self.weights[bN] = np.zeros((ly[lyr], 1))
            nnx = nx if lyr == 0 else ly[lyr - 1]
            wN = "W{}".format(lyr + 1)
            self.weights[wN] = np.random.randn(ly[lyr], nnx) * np.sqrt(2 / nnx)
