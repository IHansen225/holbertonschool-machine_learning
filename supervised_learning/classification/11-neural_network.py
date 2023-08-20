#!/usr/bin/env python3
"""
    Neural network class module
    for binary classification
"""
import numpy as np


class NeuralNetwork():
    """
        Neural network object for
        binary classification network
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # It's imp0rtant to remember that neurons are just
        # sets of matrices representing weights, activation values
        # and biases. They aren't always defined objects.
        # So, if you have a set of WBAs, you have a neuron :D
        # Obviously, you have to define all the corresponding functions
        # to make it work, but that's the idea.
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
            Returns the weights vector
            for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
            Returns the bias vector
            for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
            Returns the activation vector
            for the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
            Returns the weights vector
            for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
            Returns the bias vector
            for the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
            Returns the activation vector
            for the output neuron
        """
        return self.__A2

    def forward_prop(self, X):
        """
            Calculates the forward propagation
            of the neural network
        """
        # Same as a single neuron but f0r each layer
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        # Repeat f0r each neuron layer using the previous
        # activation vector as input
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            Calculates the cost of the
            model using logistic regression
        """
        # Yeah, it's the same as in the individual neuron
        # exercises, because the cost function works on
        # any number of neuron layers, hidden or not.
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost
