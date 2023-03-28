#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
plt.axis('off')
fig.set_figheight(12.5)
fig.set_figwidth(15)
plt.title("All in one", fontsize='x-large')
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(y0, linewidth=2.5, color='red')
ax1.set(xlim=(0, 10), xticks=np.arange(0, 11), yticks=np.arange(0, 1001, 100))
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x1, y1, color='magenta', s=7.5)
plt.title("Men's Height vs Weight", fontsize='x-small')
plt.xlabel("Height (in)", fontsize='x-small')
plt.ylabel("Weight (lbs)", fontsize='x-small')
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x2, y2)
ax3.set_yscale('log')
ax3.set(xlim=(0, 28650))
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x3, y31, 'r--', label='C-14')
ax4.plot(x3, y32, 'g-', label='Ra-226')
plt.legend()
plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')
ax4.set(xlim=(0, 20000), ylim=(0, 1))
ax5 = fig.add_subplot(gs[2, :])
ax5.hist(student_grades, bins=range(0, 110, 10), linewidth=0.5, edgecolor="black")
ax5.set(xlim=(0, 100), ylim=(0, 30), xticks=np.arange(0,100,10))
plt.title("Project A", fontsize='x-small')
plt.xlabel("Grades", fontsize='x-small')
plt.ylabel("Number of Students", fontsize='x-small')

plt.show()