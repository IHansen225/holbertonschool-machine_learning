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
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (m, h_new, w_new, c_new) = dZ.shape
    (sh, sw) = stride
    if padding == 'same':
        ph = int((((h_prev - 1) * sh) + kh - h_prev) / 2) + 1
        pw = int((((w_prev - 1) * sw) + kw - w_prev) / 2) + 1
    else:
        ph, pw = 0, 0
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)
    dA_prev = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    image_slice = A_prev_pad[i, vert_start:vert_end,
                                             horiz_start:horiz_end, :]
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c])
                    dW[:, :, :, c] += (image_slice * dZ[i, h, w, c])
                    db[:, :, :, c] += dZ[i, h, w, c]
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db