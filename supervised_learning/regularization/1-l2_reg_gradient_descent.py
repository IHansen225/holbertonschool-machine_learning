#!/usr/bin/env python3
"""
    L2 Regularization gradient
    descent module.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        Calculates one pass of gradient
        descent on the neural network.
        using the L2 regularization.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (A * (1 - A))
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
