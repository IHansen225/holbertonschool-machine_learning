#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

fig, ax = plt.subplots()
ax.plot(y, linewidth=2.5, color='red')
ax.set(xlim=(0, 10), xticks=np.arange(0, 11), yticks=np.arange(0, 1001, 100))
plt.show()
