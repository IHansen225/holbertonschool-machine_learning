#!/usr/bin/env python3
""" 
    Convolutional forward
    propagation module
    for pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """
    Convolutional foward
    propagation pass.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ch = int(((h_prev - kh) / sh) + 1)
    cw = int(((w_prev - kw) / sw) + 1)
    conv = np.zeros((m, ch, cw, c_prev))
    for i in range(ch):
        for j in range(cw):
            conv[:, i, j] = np.mean(
                (A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw]),
                axis=(1, 2),
            )
    return conv
