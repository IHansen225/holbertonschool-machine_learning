#!/usr/bin/env python3

import pandas as pd

def from_numpy(array):
    labels = [chr(65 + i) for i in range(array.shape[1])]
    dataframe = pd.DataFrame(array, columns=labels)
    return dataframe
