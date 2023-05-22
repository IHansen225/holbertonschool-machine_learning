#!/usr/bin/env python3
"""
Task 1. Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

        - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        - kernel_shape: tuple (kh, kw), containing the size of the kernel for
        the pooling
            kh: kernel height
            kw: kernel width
        - stride: tuple (sh, sw), containing the strides for the pooling
            sh: stride for the height
            sw: stride for the width
        - mode: string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """

    # Save all variables to use the necessary later:
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate the output dimensions using the pooling formula:
    output_h = int((h_prev - kh) / sh) + 1
    output_w = int((w_prev - kw) / sw) + 1

    # Create the result matrix that stores the convolved image:
    output_img = np.zeros((m, output_h, output_w, c_prev))

    # Apply the convolution:
    for i in range(output_h):
        x = i * sh

        for j in range(output_w):
            y = j * sw
            pool_slice = A_prev[:, x: x + kh, y: y + kw, :]

            # Perform pooling:
            if mode == "max":
                result = np.max(pool_slice, axis=(1, 2))
            elif mode == "avg":
                result = np.mean(pool_slice, axis=(1, 2))
            output_img[:, i, j, :] = result

    return output_img
