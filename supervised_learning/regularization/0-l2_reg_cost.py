#!/usr/bin/env python3
"""
    L2 Regularization Cost module
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Returns the L2 cost of a
        neural network
    """
    w_new = dict()
    for k, v in weights.items():
        if k[0] == 'W':
            w_new[k] = v
    res = cost + (lambtha / (2 * m)) * np.sum([np.linalg.norm(w)
                                               for w in w_new.values()])
    return res
