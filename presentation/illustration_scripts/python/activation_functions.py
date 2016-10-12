from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math

fig, ((ax)) = plt.subplots(1, 1)

x = np.arange(-5, 5, 0.1)

# Plot sigmoid
sigmoid = lambda x: 1/(1+np.exp(-x))
sig_line, = ax.plot(x, sigmoid(x), label = "Sigmoid")

# Plot ReLU
relu = lambda x: x * (x > 0)
relu_line, = ax.plot(x, relu(x), label = "ReLU")

# Scale and style plot
plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
plt.yticks(np.arange(-1, 5, 1.0))
ax.set_ylim([-0.5,2])
ax.grid(True)
ax.set_title("Common Activation Functions")

# Set legend
plt.legend(handles = [sig_line, relu_line])

plt.show()