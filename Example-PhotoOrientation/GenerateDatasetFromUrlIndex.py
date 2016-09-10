import tensorflow as tf
import urllib2
from random import shuffle, seed
from ShardingTFWriter import ShardingTFWriter
from ExampleProducer import ExampleProducer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('index_file', 'images.lst', 'output directory')
flags.DEFINE_string('dir', '.', 'output directory')
flags.DEFINE_string('prefix', 'data', 'output directory')
flags.DEFINE_integer('min_colors', 1000, 'minumum distinct colors in a usable example')
flags.DEFINE_integer('width', 256, 'example width')
flags.DEFINE_integer('height', 256, 'example height')
flags.DEFINE_integer('num', 10000, 'number of examples')
flags.DEFINE_integer('shard_size', 1000, 'maximum number of examples in a shard')


def random_image_data():
    index_file = open(FLAGS.index_file, 'r')
    urls = index_file.readlines()
    shuffle(urls)
    for url in urls:
        yield urllib2.urlopen(url).read()


def convert_to_example(image, label):
    return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }))

seed()
writer = ShardingTFWriter(FLAGS.dir, FLAGS.prefix, FLAGS.num, FLAGS.shard_size)
examples = ExampleProducer(FLAGS.width, FLAGS.height, FLAGS.min_colors, random_image_data()).example_generator()

for i in xrange(FLAGS.num):
    image_data, label = next(examples)
    example = convert_to_example(image_data, label)
    writer.write(example)
writer.close()