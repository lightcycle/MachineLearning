import tensorflow as tf


class DatasetLoader:

    @classmethod
    def __file_reader(cls, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([4], tf.int64)
            })
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image.set_shape([256, 256, 3]) # TODO: find a workaround using array_ops
        image = tf.image.resize_images(image, 100, 100)
        label = features['label']
        return image, label

    @classmethod
    def threaded_readers(cls, filenames, read_threads, num_epochs = None, shuffle = False):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs, shuffle = shuffle)
        return [cls.__file_reader(filename_queue) for _ in range(read_threads)]

    @classmethod
    def input_shuffle_batch(cls, filenames, batch_size, read_threads, num_epochs = None, min_after_dequeue = 200):
        readers = cls.threaded_readers(filenames, read_threads, num_epochs, True)
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch_join(
            readers, batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
        return image_batch, label_batch

    @classmethod
    def input_batch(cls, filenames, batch_size, read_threads, num_epochs = None):
        readers = cls.threaded_readers(filenames, read_threads, num_epochs)
        image_batch, label_batch = tf.train.batch_join(readers, batch_size = batch_size)
        return image_batch, label_batch

