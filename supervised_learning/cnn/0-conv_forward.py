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
    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    elif padding == "valid":
        ph = 0
        pw = 0

    A_prev = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), "constant", constant_values=0
    )

    h_padded = h_prev + 2 * ph
    w_padded = w_prev + 2 * pw
    ch = (h_padded - kh) // sh + 1
    cw = (w_padded - kw) // sw + 1

    conv = np.zeros((m, ch, cw, c_new))
    for i in range(ch):
        for j in range(cw):
            for k in range(c_new):
                conv[:, i, j, k] = np.sum(
                    (
                        A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw]
                        * W[:, :, :, k]
                    ),
                    axis=(1, 2, 3),
                )
    Z = conv + b
    return activation(Z)
