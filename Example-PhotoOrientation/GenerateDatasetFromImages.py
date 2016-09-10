import os
import urllib2
import tensorflow as tf
from glob import glob
from ShardingTFWriter import ShardingTFWriter
from ExampleProducer import ExampleProducer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_glob', 'images/*', 'input glob')
flags.DEFINE_string('output_dir', '.', 'output directory')
flags.DEFINE_string('prefix', 'data', 'output directory')
flags.DEFINE_integer('min_colors', 1000, 'minumum distinct colors in a usable example')
flags.DEFINE_integer('width', 256, 'example width')
flags.DEFINE_integer('height', 256, 'example height')
flags.DEFINE_integer('shard_size', 1000, 'maximum number of examples in a shard')


def filenames(glob_pattern):
    return glob(glob_pattern)


def image_data(glob_pattern):
    for filename in filenames(glob_pattern):
        url = "file://" + os.path.abspath(filename)
        yield urllib2.urlopen(url).read()


def convert_to_example(image, label):
    return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }))

writer = ShardingTFWriter(FLAGS.output_dir, FLAGS.prefix, len(filenames(FLAGS.input_glob)), FLAGS.shard_size)
examples = ExampleProducer(FLAGS.width, FLAGS.height, FLAGS.min_colors, image_data(FLAGS.input_glob)).example_generator()

for image_data, label in examples:
    example = convert_to_example(image_data, label)
    writer.write(example)
writer.close()