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
        self.__L = len(ly)
        # Checks if all the elements in ly are positive integers.
        # Lambda function to avoid using loops.
        if not len(list(filter(lambda x: x > 0, ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        # Checks if all the elements in ly are integers.
        # Same thing as above.
        if not len(list(filter(lambda x: isinstance(x, int), ly))) == self.L:
            raise TypeError("layers must be a list of positive integers")
        self.__cache = {}
        self.__weights = {}
        # Weight and bias initialization.
        # The weights are initialized using He et al. method.
        # The biases are initialized to 0.
        # The weights are initialized to a random normal distribution.
        for lyr in range(self.__L):
            bN = "b{}".format(lyr + 1)
            self.__weights[bN] = np.zeros((ly[lyr], 1))
            nnx = nx if lyr == 0 else ly[lyr - 1]
            wN = "W{}".format(lyr + 1)
            self.__weights[wN] = np.random.randn(ly[lyr], nnx)*np.sqrt(2 / nnx)

    @property
    def L(self):
        """
            Returns the number of layers
            in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
            Returns a dictionary
            with all intermediary
            values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
            Returns a dictionary
            with all weights
            and biases of the network.
        """
        return self.__weights

    def forward_prop(self, x):
        """
            Calculates the forward propagation
            of the neural network.
        """
        self.__cache["A0"] = x
        # This replicates what's done in the neural network class
        # but f0r every layer existing in the deep neural network.
        # This means that this code is modular and can be used
        # in any other neural network, no matter the number of layers.
        for lyr in range(self.__L):
            wN = "W{}".format(lyr + 1)
            bN = "b{}".format(lyr + 1)
            aN = "A{}".format(lyr + 1)
            # This line and the next go together but the damn pycodesyle
            # does not allow long lines :D
            # This is basically the sigmoid function.
            pz = np.matmul(self.weights[wN], self.cache["A{}".format(lyr)])
            z = pz + self.weights[bN]
            self.__cache[aN] = 1 / (1 + np.exp(-z))
        return self.__cache[aN], self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using
            logistic regression.
        """
        m = Y.shape[1]
        # This is the cost function.
        # It looks identical to the one in the neural network class.
        # That is because it is the same function.
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions
        """
        # Same logic as cost
        res = self.forward_prop(X)
        return np.where(res[0] >= 0.5, 1, 0), self.cost(Y, res[0])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient
            descent on the neural network.
        """
        # Same logic as cost
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y
        # The range() function goes from L to
        # 1 because it's the same backpropagation
        # algorithm. Step needs to be negative.
        for lyr in range(self.__L, 0, -1):
            wN = "W{}".format(lyr)
            bN = "b{}".format(lyr)
            aN = "A{}".format(lyr - 1)
            # Same logic as neural network class
            # but iterated over all layers.
            # Calculations are the same but because
            # we dont know how many layers there are,
            # we need to iterate over all of them.
            dw = 1 / m * np.matmul(dz, cache[aN].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(self.weights[wN].T, dz)*(cache[aN]*(1 - cache[aN]))
            # Change each weight and bias, one time
            # per iteration.
            self.__weights[wN] = self.weights[wN] - alpha * dw
            self.__weights[bN] = self.weights[bN] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the deep neural network.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for example in range(iterations):
            self.gradient_descent(Y, self.forward_prop(X)[1], alpha)
        return self.evaluate(X, Y)
