import os
import tensorflow as tf


class ShardingTFWriter:

    def __init__(self, dir, filename_prefix, total_count, shard_size, notify_every = 100):
        self.dir = dir
        self.filename_prefix = filename_prefix
        self.total_count = total_count
        self.shard_size = shard_size
        self.notify_every = notify_every
        self.count = 0
        self.writer = None
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def write(self, example):
        if self.count % self.shard_size == 0:
            if self.writer is not None:
                self.writer.close()
            path = self.__get_path()
            print "Writing to " + path
            self.writer = tf.python_io.TFRecordWriter(path)
        self.writer.write(example.SerializeToString())
        self.count += 1
        if self.count % self.notify_every == 0:
            print "Saved example " + str(self.count)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __get_path(self):
        index = self.count // self.shard_size + 1
        total_files = self.__ceil_div(self.total_count, self.shard_size)
        pattern = "{}-{:0" + str(len(str(total_files))) + "d}-of-{}.tfrecords"
        filename = pattern.format(self.filename_prefix, index, total_files)
        return os.path.join(self.dir, filename)

    @staticmethod
    def __ceil_div(num, denom):
        return (num + denom // 2) // denom