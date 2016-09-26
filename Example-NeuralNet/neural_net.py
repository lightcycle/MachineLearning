from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    scale = 1 / (x_max - x_min)
    shift = -x_min
    return (x + shift) * scale, scale, shift


def denormalize(x, scale, shift):
    return (x / scale) - shift


def bias(shape):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32))


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32))

# Data
temp_f = [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7, 71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5, 76.3]
cricket_chirps_per_s = [20, 16, 19.8, 18.4, 17.1, 15.5, 14.7, 15.7, 15.4, 16.3, 15, 17.2, 16, 17, 14.4]

# Normalize data
temp_f_normalized, x_scale, x_shift = normalize(temp_f)
cricket_chirps_per_s_normalized, y_scale, y_shift = normalize(cricket_chirps_per_s)

# Placeholders for providing data to computation graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Model (Neural Net with one hidden layer)
input_layer = tf.expand_dims(X, 1)
hidden_layer_nodes = 10
hidden_layer_weight = weight([1, hidden_layer_nodes])
hidden_layer_bias = bias([1, hidden_layer_nodes])
hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer, hidden_layer_weight) + hidden_layer_bias)
output_layer_weight = weight([hidden_layer_nodes, 1])
output_layer_bias = bias([1])
modeled_Y = tf.matmul(hidden_layer, output_layer_weight) + output_layer_bias

# Loss Function (Total Distance)
expected = tf.expand_dims(Y, 1)
loss = tf.nn.l2_loss(expected - modeled_Y)

# Training Operation
training_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as session:
    tf.initialize_all_variables().run()

    # Train model
    training_steps = 50000
    for step in range(training_steps):
        session.run([training_op, loss], feed_dict={X: temp_f_normalized, Y: cricket_chirps_per_s_normalized})

    # Plot data and trained model
    plt.plot(temp_f, cricket_chirps_per_s, 'ro', label='Data')
    inputs = np.arange(0, 1, 0.01)
    outputs = session.run(modeled_Y, feed_dict={X: inputs})
    plt.plot(denormalize(inputs, x_scale, x_shift), denormalize(outputs, y_scale, y_shift), label='Trained Model')
    plt.show()