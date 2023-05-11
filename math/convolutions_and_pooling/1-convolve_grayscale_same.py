#!/usr/bin/env python3
"""
    Performs valid same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, krn):
    """
        Returns: a numpy.ndarray containing the convolved images
        using the same convolution method.
    """
    # m is the amount of images, h*w is the image resolution
    m, h, w = images.shape
    # kh, kw is the kernel resolution
    kh, kw = krn.shape
    # output_h, output_w is the output resolution
    # generally it's h - kh + 1, w - kw + 1
    # but for same convolution, it's h and w
    output_h = h
    output_w = w
    # padding for the height, the width
    pad_h = max((kh - 1) // 2, kh // 2)
    pad_w = max((kw - 1) // 2, kw // 2)
    # pad the images with zeros on the height and width
    # in case of odd kernel (which is apparently
    # rare but it happens according to the internet)
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    # output is the output matrix, with m elements
    # of output_h * output_w size
    output = np.zeros((m, output_h, output_w))
    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # element-wise multiplication of the kernel
            output[:, i, j] = (krn * padded_images[:, i: i + kh, j: j + kw]).\
                sum(axis=(1, 2))
    return output
