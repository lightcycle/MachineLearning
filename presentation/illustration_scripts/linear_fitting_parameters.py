from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Data
temp_f = [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7, 71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5, 76.3]
chirps_per_s = [20, 16, 19.8, 18.4, 17.1, 15.5, 14.7, 15.7, 15.4, 16.3, 15, 17.2, 16, 17, 14.4]

# Known optimal parameters
optimal_weight = 0.207648
optimal_bias = 0.00260047
optimal_loss = 12.7694

# Scale
weight_scale = 0.01
bias_scale = 1
resolution = 100

# Plot loss for linear model
x = np.arange(optimal_weight - weight_scale, optimal_weight + weight_scale, weight_scale / resolution)
y = np.arange(optimal_bias - bias_scale, optimal_bias + bias_scale, bias_scale / resolution)
X, Y = np.meshgrid(x, y)
Z = np.sum((np.array(map(lambda i: i * X + Y, temp_f)) - np.array(chirps_per_s)[:,np.newaxis, np.newaxis]) ** 2, axis = 0)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('weight')
ax.set_ylabel('bias')
ax.set_zlabel('loss')
ax.scatter(optimal_weight, optimal_bias, optimal_loss, c ='r', marker ='o')
ax.plot_surface(X, Y, Z, linewidth=0.5, antialiased=True, cmap=cm.OrRd, alpha=0.5)
plt.show()

