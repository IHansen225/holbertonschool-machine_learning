#!/usr/bin/env python3
""" 
    Convolutional backward
    propagation module
    for pooling layer.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs one pass of
        backward propagation for
        pooling layers.
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (m, h_new, w_new, c_new) = dA.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    if mode == 'max':
                        image_slice = A_prev[i, vert_start:vert_end,
                                             horiz_start:horiz_end, c]
                        mask = (image_slice == np.max(image_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += (
                            mask * dA[i, h, w, c])
                    elif mode == 'avg':
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += (
                            dA[i, h, w, c] / (kh * kw))
    return dA_prev
