#!/usr/bin/env python3
""" 
    Convolutional backward
    propagation module
    for pooling layer.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """
    Performs one pass of
    backward propagation for
    pooling layers.
    """
    m, h_new, w_new, c_new = dA.shape
    m, _, _, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        mask = A_prev[i,
                                      vert_start:vert_end,
                                      horiz_start:horiz_end,
                                      c] == np.max(
                            A_prev[i,
                                   vert_start:vert_end,
                                   horiz_start:horiz_end,
                                   c]
                            )
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += mask * dA[i, h, w, c]
                    elif mode == 'avg':
                        average = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                c] += np.ones((kh, kw)) * average

    return dA_prev
