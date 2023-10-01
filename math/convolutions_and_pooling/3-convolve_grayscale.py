#!/usr/bin/env python3
"""
    Performs valid same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding="same", stride=(1, 1)):
    """
    Returns: a numpy.ndarray containing the convolved images
    using the same convolution method.
    """
    # Only note here:
    # Stride is like the step of the kernel
    # If stride is more than 1, the kernel will skip
    # some pixels and the resulting image will be smaller
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = 0, 0
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == "same":
        ph = int(((h - 1) * stride[0] + kh - h) / 2) + 1
        pw = int(((w - 1) * stride[1] + kw - w) / 2) + 1
    if ph and pw:
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
    # Stride appears as divisor of the output resolution
    # because that's how you know the size of the output
    # image.
    output_h = int(((h + (2 * ph) - kh) / stride[0]) + 1)
    output_w = int(((w + (2 * pw) - kw) / stride[1]) + 1)
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (
                kernel * images[
                    :,
                    i * stride[0]:i * stride[0] + kh,
                    j * stride[1]:j * stride[1] + kw,
                ]
            ).sum(axis=(1, 2))
    return output
