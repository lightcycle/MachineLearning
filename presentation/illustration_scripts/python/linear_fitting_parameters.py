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
weight_scale = 0.2
bias_scale = 20
resolution = 100

# Graph configuration
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('weight')
ax.set_ylabel('bias')
ax.set_zlabel('loss')

# Plot surface of loss for linear model
x = np.arange(optimal_weight - weight_scale, optimal_weight + weight_scale, weight_scale / resolution)
y = np.arange(optimal_bias - bias_scale, optimal_bias + bias_scale, bias_scale / resolution)
X, Y = np.meshgrid(x, y)
Z = np.sum((np.array(map(lambda i: i * X + Y, temp_f)) - np.array(chirps_per_s)[:,np.newaxis, np.newaxis]) ** 2, axis = 0)
ax.plot_surface(X, Y, Z, linewidth=0.5, antialiased=True, cmap=cm.OrRd, alpha=0.5)

# Plot optimizer path
x = [0, 0.26476961, 0.19365844, 0.21275727, 0.20762774, 0.20900543, 0.20863539, 0.20873477, 0.20870806, 0.20871523,
 0.20871331, 0.2087138, 0.20871365, 0.2087137, 0.20871367, 0.20871365, 0.20871367, 0.20871364, 0.20871364, 0.20871362,
 0.20871362, 0.20871361, 0.20871361, 0.20871359, 0.20871359, 0.20871358, 0.20871356, 0.20871356, 0.20871356, 0.20871353,
 0.20871355, 0.20871352, 0.20871352, 0.20871352, 0.20871349, 0.20871349, 0.20871349, 0.20871347, 0.20871347, 0.20871347,
 0.20871344, 0.20871344, 0.20871344, 0.20871341, 0.20871343, 0.20871341, 0.2087134, 0.2087134, 0.2087134, 0.20871337]
y = [0, 0.0033133335, 0.0024242869, 0.002663905, 0.0026003895, 0.0026182891, 0.0026143219, 0.0026162278, 0.0026165561,
 0.0026173082, 0.0026179466, 0.0026186153, 0.0026192761, 0.0026199392, 0.0026206013, 0.0026212637, 0.0026219264,
 0.0026225885, 0.0026232509, 0.0026239133, 0.0026245757, 0.0026252382, 0.0026259008, 0.0026265632, 0.0026272256,
 0.002627888, 0.0026285504, 0.002629213, 0.0026298754, 0.0026305376, 0.0026312002, 0.0026318624, 0.0026325251,
 0.0026331877, 0.0026338499, 0.0026345125, 0.0026351751, 0.0026358375, 0.0026364999, 0.0026371623, 0.0026378245,
 0.0026384871, 0.0026391495, 0.0026398117, 0.0026404744, 0.0026411368, 0.0026417992, 0.0026424616, 0.002643124,
 0.0026437861]
X, Y = np.meshgrid(x, y)
Z = np.sum((np.array(map(lambda i: i * X + Y, temp_f)) - np.array(chirps_per_s)[:,np.newaxis, np.newaxis]) ** 2, axis = 0)
ax.scatter(X, Y, Z)
# ax.scatter(optimal_weight, optimal_bias, optimal_loss, c ='r', marker ='o')

plt.show()

