#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_yscale('log')
ax.set(xlim=(0, 28650))
plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")

plt.show()