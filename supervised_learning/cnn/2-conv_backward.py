#!/usr/bin/env python3
""" 
    Convolutional backward
    propagation module.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Convolutional backward
    propagation pass.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2)) + 1
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2)) + 1
    if padding == "valid":
        ph = 0
        pw = 0
    A_prev = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), "constant", constant_values=0
    )
    ch = int(((h_prev + 2 * ph - kh) / sh) + 1)
    cw = int(((w_prev + 2 * pw - kw) / sw) + 1)
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))
    for i in range(ch):
        for j in range(cw):
            for k in range(c_new):
                dA_prev[:, i * sh : i * sh + kh, j * sw : j * sw + kw] += (
                    dZ[:, i, j, k].reshape(-1, 1, 1, 1) * W[:, :, :, k]
                )
                dW[:, :, :, k] += A_prev[
                    :, i * sh : i * sh + kh, j * sw : j * sw + kw
                ] * dZ[:, i, j, k].reshape(-1, 1, 1, 1)
                db[:, :, :, k] += dZ[:, i, j, k]
    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
