#!/usr/bin/env python3
"""
    Early stopping.
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    if cost > opt_cost - threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return True, count
    else:
        return False, count
