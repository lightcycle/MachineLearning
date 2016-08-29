import tensorflow as tf


class DatasetLoader:

    @staticmethod
    def __file_reader(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([4], tf.int64)
            })
        image = tf.image.decode_jpeg(features['image'], channels=3)
        # TODO set width and height in TFRecords file
        image.set_shape([100, 100, 3])
        label = features['label']
        return image, label

    def input_pipeline(self, filenames, batch_size, read_threads, num_epochs = None, min_after_dequeue = 10000):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs, shuffle = True)
        readers = [self.__file_reader(filename_queue) for _ in range(read_threads)]
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch_join(
            readers, batch_size = batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
        return image_batch, label_batch
