#!/usr/bin/env python3
"""
    L2 Regularization gradient
    descent module.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Computes one pass of gradient
        descent on a neural network
        with L2 regularization.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l-1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        dW = 1./m * np.dot(dZ, A_prev.T) + lambtha/m * W
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(W.T, dZ) * (1 - np.power(A_prev, 2))
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
