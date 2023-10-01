#!/usr/bin/env python3
"""
    Performs convolution on RGB images.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Returns: a numpy.ndarray containing the convolved images.
    """
    # Same as 3-convolve_grayscale.py but with channels
    # Basically same thing but 3D
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    ph, pw = 0, 0
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = int(((h - 1) * stride[0] + kh - h) / 2) + 1
        pw = int(((w - 1) * stride[1] + kw - w) / 2) + 1
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    output_h = int(((h + (2 * ph) - kh) / stride[0]) + 1)
    output_w = int(((w + (2 * pw) - kw) / stride[1]) + 1)
    output = np.zeros((m, output_h, output_w, nc))
    for x in range(nc):
        kernel = kernels[:, :, :, x]
        for i in range(output_h):
            for j in range(output_w):
                output[:, i, j, x] = (
                    kernel * images[
                        :,
                        i * stride[0]: i * stride[0] + kh,
                        j * stride[1]: j * stride[1] + kw,
                        :
                    ]
                ).sum(axis=(1, 2, 3))
    return output
