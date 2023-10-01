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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ch = int(((h_prev - kh) / sh) + 1)
    cw = int(((w_prev - kw) / sw) + 1)
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    for i in range(ch):
        for j in range(cw):
            if mode == "max":
                dA_prev[:, i * sh : i * sh + kh, j * sw : j * sw + kw] += (
                    dA[:, i, j].reshape(-1, 1, 1, 1)
                    * A_prev[:, i * sh : i * sh + kh, j * sw : j * sw + kw]
                )
            if mode == "avg":
                dA_prev[:, i * sh : i * sh + kh, j * sw : j * sw + kw] += dA[
                    :, i, j
                ].reshape(-1, 1, 1, 1) / (kh * kw)
    return dA_prev
