import tensorflow as tf
from operator import mul


class Model:

    @classmethod
    def __weight_variable(cls, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @classmethod
    def __bias_variable(cls, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @classmethod
    def __conv_layer(cls, input_shape, input, k_width, k_height, k_depth, num_outputs):
        weight = cls.__weight_variable([k_width, k_height, k_depth, num_outputs])
        bias = cls.__bias_variable([num_outputs])
        return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME') + bias,\
               (input_shape[0], input_shape[1], num_outputs)

    @classmethod
    def __relu_layer(cls, input_shape, input):
        return tf.nn.relu(input),\
               input_shape

    @classmethod
    def __maxpool_layer(cls, input_shape, input, pool_size):
        return tf.nn.max_pool(input, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME'),\
               (input_shape[0] / pool_size, input_shape[1] / pool_size, input_shape[2])

    @classmethod
    def __fully_connected_layer(cls, input_shape, input, num_outputs):
        flattened_size = reduce(mul, input_shape, 1)
        weight = cls.__weight_variable([flattened_size, num_outputs])
        bias = cls.__bias_variable([num_outputs])
        input_flat = tf.reshape(input, [-1, flattened_size])
        return tf.matmul(input_flat, weight) + bias,\
               (num_outputs,)

    @classmethod
    def __dropout(cls, input_shape, input, keep_prob):
        return tf.nn.dropout(input, keep_prob), \
               input_shape

    @classmethod
    def __softmax(cls, input_shape, model):
        return tf.nn.softmax(model), \
               input_shape

    @classmethod
    def create_graph(cls, input, keep_prob):
        # Normalize [0,255] ints to [0,1] floats
        input_float = tf.div(tf.cast(input, tf.float32), 255)

        # Build model
        model = input_float
        shape = (256, 256, 3)
        model, shape = cls.__conv_layer(shape, model, 3, 3, 3, 4)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__maxpool_layer(shape, model, 2)
        model, shape = cls.__conv_layer(shape, model, 3, 3, 4, 8)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__maxpool_layer(shape, model, 2)
        model, shape = cls.__conv_layer(shape, model, 3, 3, 8, 16)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__maxpool_layer(shape, model, 2)
        model, shape = cls.__conv_layer(shape, model, 3, 3, 16, 32)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__maxpool_layer(shape, model, 2)
        model, shape = cls.__conv_layer(shape, model, 3, 3, 32, 64)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__maxpool_layer(shape, model, 2)
        model, shape = cls.__fully_connected_layer(shape, model, 128)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__dropout(shape, model, keep_prob)
        model, shape = cls.__fully_connected_layer(shape, model, 4)
        model, shape = cls.__relu_layer(shape, model)
        model, shape = cls.__softmax(shape, model)

        return model
