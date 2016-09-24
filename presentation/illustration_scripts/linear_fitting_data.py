from __future__ import division
import matplotlib.pyplot as plt

# Data
temp_f = [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7, 71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5, 76.3]
chirps_per_s = [20, 16, 19.8, 18.4, 17.1, 15.5, 14.7, 15.7, 15.4, 16.3, 15, 17.2, 16, 17, 14.4]

# Plot data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Temp (F)')
ax.set_ylabel('Cricket Chirps / sec')
ax.plot(temp_f, chirps_per_s, 'ro', label='Data')
plt.show()
