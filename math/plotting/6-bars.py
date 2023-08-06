#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fig, ax = plt.subplots()
x_labels = ['Farrah', 'Fred', 'Felicia']
x_pos = np.arange(len(x_labels))
y_limits = (0, 80)
y_ticks = np.arange(0, 81, 10)
for i in range(fruit.shape[0]):
    ax.bar(x_pos, fruit[i], bottom=np.sum(fruit[:i], axis=0), color=colors[i], label=['Apples', 'Bananas', 'Oranges', 'Peaches'][i], width=0.5)
ax.legend()
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(y_limits)
ax.set_yticks(y_ticks)
ax.set_title('Number of Fruit per Person')
plt.show()