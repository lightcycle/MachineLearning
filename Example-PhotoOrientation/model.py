import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, size):
  return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def create_graph(input_image_size):
    # Define placeholder for input image data
    x = tf.placeholder(tf.float32, [None, input_image_size[0], input_image_size[1], 3])

    # Normalize [0,255] ints to [0,1] floats
    x_float = tf.div(tf.cast(x, tf.float32), 255)

    # First layer group: convolutional -> reLU -> max pooling
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_float, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, 5)

    # Second layer group: convolutional -> reLU -> max pooling
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, 4)

    # Fully connected layer
    W_fc1 = weight_variable([10 * 10 * 64, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Loss layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([128, 4])
    b_fc2 = bias_variable([4])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return x, y_conv, keep_prob, h_conv1, b_conv1, h_conv2, b_conv2
