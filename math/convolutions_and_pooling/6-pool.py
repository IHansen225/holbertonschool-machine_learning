#!/usr/bin/env python3
"""
    Performs pooling on RGB images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Returns: a numpy.ndarray containing the convolved images.
    """
    # Same as 3-convolve_grayscale.py but with channels
    # Basically same thing but 3D
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    output_h = int((h - kh) / stride[0]) + 1
    output_w = int((w - kw) / stride[1]) + 1
    output = np.zeros((m, output_h, output_w, c))
    for i in range(output_h):
        for j in range(output_w):
            k = images[:, i * stride[0]: i * stride[0] +
                       kh, j * stride[1]: j * stride[1] + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(k, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(k, axis=(1, 2))
    return output
