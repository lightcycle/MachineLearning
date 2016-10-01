from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math

fig, ((ax)) = plt.subplots(1, 1)

x = np.arange(-5, 5, 0.1)

# Plot sigmoids
sigmoid = lambda x, weight, bias, output_weight: output_weight/(1+np.exp(x * weight + bias))
sig_line1, = ax.plot(x, sigmoid(x, 2, 4, 2), label = "Sigmoid 1")
sig_line2, = ax.plot(x, sigmoid(x, 1, -5, 3), label = "Sigmoid 2")
sig_line3, = ax.plot(x, sigmoid(x, -2, 4, 2), label = "Sigmoid 3")
combined, = ax.plot(x, sigmoid(x, 2, 4, 2) + sigmoid(x, 1, -5, 3) + sigmoid(x, -2, 4, 2), label = "Combined")

# Scale and style plot
# plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
# plt.yticks(np.arange(-1, 5, 1.0))
ax.set_ylim([-1,6])
ax.grid(True)

# Set legend
plt.legend(handles = [sig_line1, sig_line2, sig_line3, combined], loc = 3)

plt.show()