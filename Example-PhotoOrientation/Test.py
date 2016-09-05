import tensorflow as tf
from TFRunner import TFRunner
import multiprocessing
from glob import glob
from DatasetLoader import DatasetLoader
from Model import Model
from Average import Average

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_string('test_files_glob', './input/test*.tfrecords', 'glob for TFRecords files containing testing data')
flags.DEFINE_string('model_file', './model.ckpt', 'path to save or load trained model parameters')
flags.DEFINE_integer('read_threads', multiprocessing.cpu_count(), 'number of reading threads')
flags.DEFINE_string('profile', 'trace_test.json', 'a Chrome trace file will be written at the specified path for the first training batch')
flags.DEFINE_string('summary', './tensorboard_test', 'Tensorboard output directory')

# Testing input
dataset_loader = DatasetLoader()
keep_prob_holder = tf.placeholder(tf.float32, shape = ())
image_batch, label_batch = dataset_loader.input_batch(
    glob(FLAGS.test_files_glob), FLAGS.batch_size, FLAGS.read_threads, num_epochs = 2)
label_batch = tf.cast(label_batch, tf.float32)

# Model, correctness predicate, and correctness aggregator
inferred_labels = Model.create_graph(image_batch, keep_prob_holder)
correct_prediction = tf.equal(tf.argmax(inferred_labels, 1), tf.argmax(tf.cast(label_batch, tf.float32), 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run graph
average = Average()
TFRunner.run(
    accuracy_op,
    feed_dict = {keep_prob_holder: 1.0},
    restore_checkpoint = FLAGS.model_file,
    profile = FLAGS.profile,
    summary = FLAGS.summary,
    batch_result_callback = average.add
)
print 'Model accuracy: %g' % average.calculate()