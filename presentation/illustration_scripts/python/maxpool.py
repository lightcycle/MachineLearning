import tensorflow as tf
import numpy as np
from scipy.misc import imsave


# Inputs
image_data_placeholder = tf.placeholder(tf.string)
image = tf.image.decode_png(image_data_placeholder, channels=1)
image.set_shape([8, 8, 1])
image_batch = tf.expand_dims(image, 0)
input_float = tf.div(tf.cast(image_batch, tf.float32), 255)

# Start TensorFlow session
sess = tf.Session()

# Run model
image_file = open('sample_small_input_bw.png', 'r')
image_data = image_file.read()
pool_op = tf.nn.max_pool(input_float, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pool_out = sess.run([pool_op], feed_dict = {image_data_placeholder: image_data})
output = np.array(pool_out)[0,0,...,0]
imsave("maxpool.png", output)

# Close TensorFlow session
sess.close()
