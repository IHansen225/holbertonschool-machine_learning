#!/usr/bin/env python3
"""
    Neural network class module
    for binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions
        """
        # Same logic as cost
        res = self.forward_prop(X)
        return np.where(res[1] >= 0.5, 1, 0), self.cost(Y, res[1])

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent
            on the neural network
        """
        # Same logic as cost but backwards
        # bc you have to change the weights and biases
        # from the output to the back (input)
        # That's why it's called backpropagation :D
        m = Y.shape[1]
        # Output layer calculations
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        # Hidden layer calculations
        # This part of the code is modular and
        # can be used f0r any number of hidden layers
        # configured correctly
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
            Trains the neural network over
            a specified number of iterations
        """
        # Same logic as single neuron but with one
        # activation vector per layer
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            step_list = np.arange(0, iterations + 1, step)
            step_cost = []
        for i in range(iterations):
            cost = self.cost(Y, self.__A2)
            if graph:
                step_cost.append(cost)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        if graph:
            fig, ax = plt.subplots()
            ax.plot(step_cost, linewidth=2.5, color='red')
            ax.set(xlim=iterations + 1, xticks=step_list)
            plt.show()
        return self.evaluate(X, Y)
