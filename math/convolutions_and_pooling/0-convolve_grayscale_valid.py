#!/usr/bin/env python3
"""
    Performs valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        Returns: a numpy.ndarray containing the convolved images.
    """
    # m is the amount of images, h*w is the image resolution
    m, h, w = images.shape
    # kh, kw is the kernel resolution
    kh, kw = kernel.shape
    # output_h, output_w is the output resolution
    # generally it's h - kh + 1, w - kw + 1
    output_h = h - kh + 1
    output_w = w - kw + 1
    # output is the output matrix, with m elements
    # of output_h * output_w size
    output = np.zeros((m, output_h, output_w))
    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # element-wise multiplication of the kernel
            output[:, i, j] = (kernel * images[:, i: i + kh, j: j + kw]).\
                sum(axis=(1, 2))
    return output
