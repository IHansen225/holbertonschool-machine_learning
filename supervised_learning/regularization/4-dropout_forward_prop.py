#!/usr/bin/env python3
"""
    Dropout Forward Propagation
    module.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using Dropout.
    """
    aux = {}
    A = X
    for l in range(1, L):
        A_prev = A
        Z = np.dot(weights['W' + str(l)], A_prev) + weights['b' + str(l)]
        A = np.tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A *= D
        A /= keep_prob
        aux['Z' + str(l)] = Z
        aux['D' + str(l)] = D
        aux['A' + str(l)] = A
    Z = np.dot(weights['W' + str(L)], A) + weights['b' + str(L)]
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    aux['Z' + str(L)] = Z
    aux['A' + str(L)] = A
    return aux
