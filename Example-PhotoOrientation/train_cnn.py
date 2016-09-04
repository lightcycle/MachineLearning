import tensorflow as tf
from tensorflow.python.client import timeline
import multiprocessing
import os
import shutil
from glob import glob
from DatasetLoader import DatasetLoader
from Model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epochs', 30, 'number of times to run through training dataset')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_string('train_files_glob', './input/train*.tfrecords', 'glob for TFRecords files containing training data')
flags.DEFINE_string('test_files_glob', './input/test*.tfrecords', 'glob for TFRecords files containing testing data')
flags.DEFINE_string('model_file', './model.ckpt', 'path to save or load trained model parameters')
flags.DEFINE_boolean('train', True, 'whether or not to train model')
flags.DEFINE_boolean('test', True, 'whether or not to test model')
flags.DEFINE_integer('read_threads', multiprocessing.cpu_count(), 'number of reading threads')
flags.DEFINE_string('profile_training_batch', 'trace.json', 'if set, a Chrome trace file will be written at the specified path for the first training batch')

dataset_loader = DatasetLoader()

def recreate(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

image_batch_holder = tf.placeholder(tf.uint8, shape = (None, 200, 200, 3))
label_batch_holder = tf.placeholder(tf.int64, shape = (None, 4))
keep_prob_holder = tf.placeholder(tf.float32, shape = ())

inferred_labels = Model.create_graph(image_batch_holder, keep_prob_holder)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(label_batch_holder, tf.float32) * tf.log(inferred_labels), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(inferred_labels, 1), tf.argmax(tf.cast(label_batch_holder, tf.float32), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


def trace_options(trace_path):
    if trace_path is not None:
        return tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), tf.RunMetadata()
    else:
        return None, None


with tf.Session() as sess:

    if FLAGS.train:
        print "Starting training."

        image_batch, label_batch = dataset_loader.input_shuffle_batch(
            glob(FLAGS.train_files_glob), FLAGS.batch_size, FLAGS.read_threads, num_epochs=FLAGS.training_epochs)
        label_batch = tf.cast(label_batch, tf.float32)

        sess.run(tf.initialize_all_variables())

        writer = tf.train.SummaryWriter("./tensorboard", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image_batch_eval = image_batch.eval()
        label_batch_eval = label_batch.eval()

        try:
            while not coord.should_stop():
                run_options, run_metadata = trace_options(FLAGS.profile_training_batch)
                sess.run(train_step, feed_dict = {
                    image_batch_holder: image_batch_eval,
                    label_batch_holder: label_batch_eval,
                    keep_prob_holder: 0.5
                }, options = run_options, run_metadata = run_metadata)
                writer.add_summary(sess.run(tf.merge_all_summaries()))
                if FLAGS.profile_training_batch is not None:
                    with open(FLAGS.profile_training_batch, 'w') as f:
                        f.write(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
                    FLAGS.profile_training_batch = None
                print 'Finished batch'
        except tf.errors.OutOfRangeError:
            print('Training complete -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        writer.close()

        print "Saving model."
        save_path = saver.save(sess, FLAGS.model_file)
        print "Model saved."
    else:
        print "Restoring previously trained model."
        saver.restore(sess, FLAGS.model_file)
        print "Model restored."

    if FLAGS.test:
        print "Starting test."

        image_batch, label_batch = dataset_loader.input_batch(
            glob(FLAGS.test_files_glob), FLAGS.batch_size, FLAGS.read_threads, num_epochs=FLAGS.training_epochs)
        label_batch = tf.cast(label_batch, tf.float32)

        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image_batch_eval = image_batch.eval()
        label_batch_eval = label_batch.eval()

        scores = list()
        try:
            while not coord.should_stop():
                score = sess.run(accuracy, feed_dict={
                    image_batch_holder: image_batch_eval,
                    label_batch_holder: label_batch_eval,
                    keep_prob_holder: 1
                })
                print("\ttest accuracy %g" % score)
                scores.append(score)
            print "Overall test accuracy %g" % (reduce(lambda x, y: x + y, scores) / len(scores))

        except tf.errors.OutOfRangeError:
            print('Testing complete -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
