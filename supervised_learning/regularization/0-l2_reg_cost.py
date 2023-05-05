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
    res = cost + (lambtha / (2 * m)) * np.sum([np.linalg.norm(w)
                                               for w in weights.values()
                                               if w is not weights['b' + str(L)]])
    return res
