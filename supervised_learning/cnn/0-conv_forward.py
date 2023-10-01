#!/usr/bin/env python3
""" 
    Convolutional Forward
    propagation module.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        Performs one pass of
        forward propagation
        over a convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    output_h = int((h_prev + 2 * ph - kh) / sh) + 1
    output_w = int((w_prev + 2 * pw - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w, c_new))
    padA = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                output[:, i, j, k] = np.sum(padA[:, i * sh: i * sh + kh,
                                                 j * sw: j * sw + kw] *
                                           W[:, :, :, k], axis=(1, 2, 3))
    output = output + b
    return activation(output)
