#!/usr/bin/env python3
"""
Task 3. Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

        - dA: numpy.ndarray (m, h_new, w_new, c_new) containing the partial
        derivatives with respect to the unactivated output of the convolutional
        layer
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c_new: number of channels
        - A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer
            h_prev: the height of the previous layer
            w_prev: the width of the previous layer
            c: the number of channels
        - kernel_shape: tuple (kh, kw) containing the size of the kernel for
        the pooling
            kh is the kernel height
            kw is the kernel width
        - stride: tuple of (sh, sw) containing the strides for the convolution
            sh: stride for the height
            sw: stride for the width
        - mode: string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively

    Return: the partial derivatives with respect to the prev layer (dA_prev)
    """

    # Save all variables to use the necessary later:
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Create the result matrix:
    dA_prev = np.zeros_like(A_prev)

    # Perform backpropagation:
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    da = dA[i, h, w, c]

                    # Get the corners of the current slice:
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Calculate the gradients using the pooling mode:
                    if mode == "max":
                        # Find the position of the maximum value in the window:
                        pool = A_prev[i, v_start:v_end, h_start:h_end, c]
                        mask = (pool == np.max(pool))
                    elif mode == "avg":
                        # Compute the average value of the window:
                        mask = np.ones((kh, kw)) / (kh * kw)
                    else:
                        return

                    dA_prev[i, v_start:v_end, h_start:h_end, c] += (mask * da)

    return dA_prev
