import tensorflow as tf
from glob import glob
from dataset_loader import load_dataset

x = tf.placeholder(tf.float32, [None, 50, 50, 3])

y_ = tf.placeholder(tf.float32, [None, 4])

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

x_float = tf.div(tf.cast(x, tf.float32), 255)

W_conv1 = weight_variable([5, 5, 3, 8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x_float, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 5)

W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2)

W_fc1 = weight_variable([5 * 5 * 16, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 4])
b_fc2 = bias_variable([4])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_file_list = glob("./train/*.jpg")

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(100000):
    train_images, train_labels = load_dataset(training_file_list, batch_size = 50, scale = (50, 50))
    if i%50 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})

test_images, test_labels = load_dataset(glob("./test/*.jpg"), scale = (50, 50))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_images, y_: test_labels, keep_prob: 1.0}))