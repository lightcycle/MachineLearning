import tensorflow as tf
import numpy as np
from scipy.misc import imsave


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Inputs
image_data_placeholder = tf.placeholder(tf.string)
image = tf.image.decode_jpeg(image_data_placeholder, channels=3)
image.set_shape([256, 256, 3])
image_batch = tf.expand_dims(image, 0)

# Model
input_float = tf.div(tf.cast(image_batch, tf.float32), 255)
conv_relu_pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(input_float,      weight_variable([3, 3,  3, 16]), strides=[1, 1, 1, 1], padding='SAME') + bias_variable([16])), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv_relu_pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(conv_relu_pool_1, weight_variable([3, 3, 16, 32]), strides=[1, 1, 1, 1], padding='SAME') + bias_variable([32])), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv_relu_pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(conv_relu_pool_2, weight_variable([3, 3, 32, 48]), strides=[1, 1, 1, 1], padding='SAME') + bias_variable([48])), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv_relu_pool_4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(conv_relu_pool_3, weight_variable([3, 3, 48, 64]), strides=[1, 1, 1, 1], padding='SAME') + bias_variable([64])), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv_relu_pool_5 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(conv_relu_pool_4, weight_variable([3, 3, 64, 80]), strides=[1, 1, 1, 1], padding='SAME') + bias_variable([80])), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
flat = tf.reshape(conv_relu_pool_5, [-1, 8 * 8 * 80])
fc1_relu = tf.nn.relu(tf.matmul(flat, weight_variable([8 * 8 * 80, 160])) + bias_variable([160]))
fc2 = tf.matmul(tf.reshape(fc1_relu, [-1, 160]), weight_variable([160, 4])) + bias_variable([4])
softmax = tf.nn.softmax(fc2)

# Start TensorFlow session
sess = tf.Session()

# Restore saved parameters
saver = tf.train.Saver(sharded=True)
saver.restore(sess, '../../Example-PhotoOrientation/train/model.ckpt')

# Run model
image_file = open('sample_input.jpg', 'r')
image_data = image_file.read()
conv_relu_pool_1_out, conv_relu_pool_2_out, conv_relu_pool_3_out, conv_relu_pool_4_out, conv_relu_pool_5_out, flat_out, fc1_relu_out, fc2_out, softmax_out = sess.run([conv_relu_pool_1, conv_relu_pool_2, conv_relu_pool_3, conv_relu_pool_4, conv_relu_pool_5, flat, fc1_relu, fc2, softmax], feed_dict = {image_data_placeholder: image_data})

# Output details
num_layer = 1
for image_layer in [conv_relu_pool_1_out, conv_relu_pool_2_out, conv_relu_pool_3_out, conv_relu_pool_4_out, conv_relu_pool_5_out]:
    layer = np.squeeze(image_layer)
    for index in range(layer.shape[2]):
        image = layer[...,index]
        imsave("layer" + str(num_layer) + "_output" + str(index) + ".png", image)
    num_layer += 1
print flat_out.shape
print flat_out
print fc1_relu_out.shape
print fc1_relu_out
print fc2_out.shape
print fc2_out
print softmax_out.shape
print softmax_out

# Close TensorFlow session
sess.close()
