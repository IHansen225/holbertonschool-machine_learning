#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(7.5)
ax.hist(student_grades, bins=range(0, 110, 10), linewidth=0.5, edgecolor="black")
ax.set(xlim=(0, 100), ylim=(0, 30), xticks=np.arange(0,100,10))
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

plt.show()
