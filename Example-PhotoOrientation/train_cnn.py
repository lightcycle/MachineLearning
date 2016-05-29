import tensorflow as tf
import operator
from glob import glob
from dataset_loader import load_dataset, load_image
from itertools import compress
from scipy.misc import imsave

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('train_steps', 10000, 'number of steps to run trainer')
flags.DEFINE_integer('update_steps', 100, 'interval at which to print step count with current accuracy')
flags.DEFINE_integer('batch_size', 50, 'batch size')
flags.DEFINE_string('train_dir', './train', 'directory containing training data')
flags.DEFINE_string('test_dir', './test', 'directory containing training data')
flags.DEFINE_string('model_file', './model.ckpt', 'path to save or load trained model parameters')
flags.DEFINE_boolean('train', True, 'whether or not to train model')
flags.DEFINE_boolean('test', True, 'whether or not to test model')
flags.DEFINE_boolean('save_test_examples', False, 'whether or not to save examples of passing and failing model input')
flags.DEFINE_integer('num_passing_test_examples', 10, 'number of examples of passing model input to save')
flags.DEFINE_integer('num_failing_test_examples', 10, 'number of examples of failing model input to save')
flags.DEFINE_string('examine_file', None, 'path to a specific image input to save detailed results for')

image_size = (100, 100)

x = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3])
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

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_float, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 5)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2)

W_fc1 = weight_variable([10 * 10 * 64, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if FLAGS.train:
        print "Starting training."
        training_file_list = glob(FLAGS.train_dir + "/*.jpg")
        for i in range(FLAGS.train_steps):
            train_images, train_labels = load_dataset(training_file_list, batch_size = FLAGS.batch_size, scale = image_size)
            if i > 0 and i % (FLAGS.update_steps) == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                print("\tstep %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})
        print "Training complete, saving model."
        save_path = saver.save(sess, FLAGS.model_file)
        print "Model saved."
    else:
        print "Restoring previously trained model."
        saver.restore(sess, FLAGS.model_file)
        print "Model restored."

    if FLAGS.test:
        print "Loading test dataset."
        test_images, test_labels = load_dataset(glob(FLAGS.test_dir + "/*.jpg"), scale = image_size)
        print "Starting test."
        scores = list()
        for i in xrange(0, len(test_images), FLAGS.batch_size):
            score = accuracy.eval(session=sess, feed_dict={x: test_images[i:i+FLAGS.batch_size], y_: test_labels[i:i+FLAGS.batch_size], keep_prob: 1.0})
            print("\ttest accuracy %g" % score)
            scores.append(score)
        print "Overall test accuracy %g" % (reduce(lambda x, y: x + y, scores) / len(scores))


    if FLAGS.save_test_examples:
        pass_count = FLAGS.num_passing_test_examples
        fail_count = FLAGS.num_passing_test_examples
        while pass_count > 0 and fail_count > 0:
            test_images, test_labels = load_dataset(glob(FLAGS.test_dir + "/*.jpg") , batch_size = FLAGS.batch_size, scale = image_size)
            correct = correct_prediction.eval(session=sess, feed_dict={
                x: test_images, y_: test_labels, keep_prob: 1.0})
            for pass_image in compress(test_images, correct):
                if pass_count > 0:
                    imsave("passed_" + str(pass_count) + ".jpg", pass_image)
                    pass_count -= 1
            for fail_image in compress(test_images, map(operator.not_, correct)):
                if fail_count > 0:
                    imsave("failed_" + str(fail_count) + ".jpg", fail_image)
                    fail_count -= 1

    if FLAGS.examine_file:
        image = load_image(FLAGS.examine_file)
        images = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])
        print images.shape
