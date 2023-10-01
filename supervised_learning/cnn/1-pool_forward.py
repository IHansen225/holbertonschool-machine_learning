#!/usr/bin/env python3
""" 
    Convolutional forward
    propagation module
    for pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): 
    """
        Convolutional foward
        propagation pass.
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    n_h = int(((h_prev - kh) / sh) + 1)
    n_w = int(((w_prev - kw) / sw) + 1)
    output = np.zeros((m, n_h, n_w, c_prev))
    for i in range(n_h):
        for j in range(n_w):
            for k in range(c_prev):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                image_slice = A_prev[:, vert_start:vert_end,
                                     horiz_start:horiz_end, k]
                if mode == 'max':
                    output[:, i, j, k] = np.max(image_slice, axis=(1, 2))
                elif mode == 'avg':
                    output[:, i, j, k] = np.mean(image_slice, axis=(1, 2))
    return output
