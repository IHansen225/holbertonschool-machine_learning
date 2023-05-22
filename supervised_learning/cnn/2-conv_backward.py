#!/usr/bin/env python3
"""
Task 2. Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

        - dZ: numpy.ndarray (m, h_new, w_new, c_new) containing the partial
        derivatives with respect to the unactivated output of the convolutional
        layer
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c_new: number of channels in the output
        - A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer
            h_prev: the height of the previous layer
            w_prev: the width of the previous layer
            c_prev: the number of channels in the previous layer
        - W: numpy.ndarray (kh, kw, c_prev, c_new) containing the kernels for
        the convolution
            kh: filter height
            kw: filter width
        - b: numpy.ndarray (1, 1, 1, c_new) containing the biases applied to
        the convolution
        - padding: string that is either same or valid, indicating the type of
        padding used
        - stride: tuple of (sh, sw) containing the strides for the convolution
            sh: stride for the height
            sw: stride for the width

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """

    # Save all variables to use the necessary later:
    m_dZ, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = W.shape[0], W.shape[1]
    c_new_bias = b.shape[3]
    sh, sw = stride

    # Set the images padding:
    if padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = int((((h_prev - 1) * sh) + kh - h_prev) / 2 + 1)
        pad_w = int((((w_prev - 1) * sw) + kw - w_prev) / 2 + 1)
    else:
        return

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                        (0, 0)), mode="constant")

    # Initialize the gradients:
    dA_prev = np.zeros((m, h_prev + (2 * pad_h), w_prev + (2 * pad_w), c_prev))
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Compute gradients:
    for i in range(m):
        for k_idx in range(c_new):
            for j in range(h_new):
                for k in range(w_new):
                    h, w = (j * sh), (k * sw)

                    dA_prev[i, h: h + kh, w: w + kw, :] += (
                        dZ[i, j, k, k_idx] * W[:, :, :, k_idx]
                        )

                    dW[:, :, :, k_idx] += (
                        A_prev_pad[i, h: h + kh, w: w + kw, :] *
                        dZ[i, j, k, k_idx]
                        )

    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA_prev, dW, db
