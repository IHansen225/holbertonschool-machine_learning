#!/usr/bin/env python3
"""
Task 0. Convolutional Forward Prop
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

        - A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        - W: numpy.ndarray (kh, kw, c_prev, c_new) containing the kernels for
        the convolution
            kh: filter height
            kw: filter width
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        - b: numpy.ndarray (1, 1, 1, c_new) containing the biases applied to
        the convolution
        - activation: activation function applied to the convolution
        - padding: string that is either same or valid, indicating the type of
        padding used
        - stride: tuple (sh, sw) containing the strides for the convolution

    Returns: the output of the convolutional layer
    """

    # Save all variables to use the necessary later:
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Set the images padding:
    if padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = int((((h_prev - 1) * sh) + kh - h_prev) / 2)
        pad_w = int((((w_prev - 1) * sw) + kw - w_prev) / 2)
    else:
        return

    img_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                     "constant", constant_values=0)

    # Calculate output dimensions:
    output_h = int((h_prev + 2 * pad_h - kh) / sh + 1)
    output_w = int((w_prev + 2 * pad_w - kw) / sw + 1)

    # Create the result matrix that stores the convolved image:
    output_img = np.zeros((m, output_h, output_w, c_new))

    # Apply the convolution:
    for i in range(output_h):
        x = sh * i

        for j in range(output_w):
            y = sw * j
            img_slice = img_pad[:, x: x + kh, y: y + kw, :]

            for k in range(c_new):
                output_img[:, i, j, k] = np.sum(
                                    img_slice * W[:, :, :, k], axis=(1, 2, 3))

    return activation(output_img + b)
