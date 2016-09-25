import tensorflow as tf
import numpy as np
from scipy.misc import imsave


# Inputs
image_data_placeholder = tf.placeholder(tf.string)
kernal_placeholder = tf.placeholder(tf.float32, [3, 3, 3, 1])
image = tf.image.decode_jpeg(image_data_placeholder, channels=3)
image.set_shape([256, 256, 3])
image_batch = tf.expand_dims(image, 0)
input_float = tf.div(tf.cast(image_batch, tf.float32), 255)

# Kernals (depth is y-axis)
kernals = {
    "red": tf.expand_dims(tf.transpose(tf.constant(
        [
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ], tf.float32), perm=[1,2,0]), 3),
    "blue": tf.expand_dims(tf.transpose(tf.constant(
        [
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
        ], tf.float32), perm=[1,2,0]), 3),
    "gaussian_blur": tf.expand_dims(tf.transpose(tf.constant(
        [
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]],
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]],
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]
        ], tf.float32), perm=[1, 2, 0]), 3),
    "edge": tf.expand_dims(tf.transpose(tf.constant(
        [
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]],
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]],
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]]
        ], tf.float32), perm=[1, 2, 0]), 3)
}

# Start TensorFlow session
sess = tf.Session()

# Run model
image_file = open('sample_input.jpg', 'r')
image_data = image_file.read()
for name, kernal in kernals.items():
    kernal_op = tf.nn.relu(tf.nn.conv2d(input_float, kernal, strides=[1, 1, 1, 1], padding='SAME'))
    kernal_out = sess.run([kernal_op], feed_dict = {image_data_placeholder: image_data})
    output = np.array(kernal_out)[0,0,...,0]
    imsave(name + ".png", output)

# Close TensorFlow session
sess.close()
