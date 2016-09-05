import tensorflow as tf
from TFRunner import TFRunner
import multiprocessing
from glob import glob
from DatasetLoader import DatasetLoader
from Model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epochs', 30, 'number of times to run through training dataset')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_string('train_files_glob', './input/train*.tfrecords', 'glob for TFRecords files containing training data')
flags.DEFINE_string('model_file', './model.ckpt', 'path to save or load trained model parameters')
flags.DEFINE_integer('read_threads', multiprocessing.cpu_count(), 'number of reading threads')
flags.DEFINE_string('profile', 'trace_train.json', 'a Chrome trace file will be written at the specified path for the first training batch')
flags.DEFINE_string('summary', './tensorboard_train', 'Tensorboard output directory')

# Training input
dataset_loader = DatasetLoader()
keep_prob_holder = tf.placeholder(tf.float32, shape = ())
image_batch, label_batch = dataset_loader.input_shuffle_batch(
    glob(FLAGS.train_files_glob), FLAGS.batch_size, FLAGS.read_threads, num_epochs = FLAGS.training_epochs)
label_batch = tf.cast(label_batch, tf.float32)

# Model, loss function, and training op
inferred_labels = Model.create_graph(image_batch, keep_prob_holder)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch, tf.float32) * tf.log(inferred_labels), reduction_indices=[1]))
training_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Run graph
TFRunner.run(
    training_op,
    feed_dict = {keep_prob_holder: 0.5},
    save_checkpoint = FLAGS.model_file,
    profile = FLAGS.profile,
    summary = FLAGS.summary
)