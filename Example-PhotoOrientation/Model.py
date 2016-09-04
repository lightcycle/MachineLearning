import tensorflow as tf


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
    def __conv2d(cls, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @classmethod
    def __max_pool(cls, x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    @classmethod
    def __conv_layer(cls, input_shape, input, k_width, k_height, k_depth, num_outputs, max_pool):
        W_conv1 = cls.__weight_variable([k_width, k_height, k_depth, num_outputs])
        b_conv1 = cls.__bias_variable([num_outputs])
        h_conv1 = tf.nn.relu(Model.__conv2d(input, W_conv1) + b_conv1)
        return Model.__max_pool(h_conv1, max_pool), (input_shape[0] / max_pool, input_shape[1] / max_pool, num_outputs)

    @classmethod
    def __fully_connected_layer(cls, input_shape, input, num_outputs):
        W_fc1 = cls.__weight_variable([input_shape[0] * input_shape[1] * input_shape[2], num_outputs])
        b_fc1 = cls.__bias_variable([num_outputs])
        conv_flat = tf.reshape(input, [-1, input_shape[0] * input_shape[1] * input_shape[2]])
        return tf.nn.relu(tf.matmul(conv_flat, W_fc1) + b_fc1), num_outputs

    @classmethod
    def __loss_layer(cls, input_shape, input, keep_prob, num_outputs):
        h_fc1_drop = tf.nn.dropout(input, keep_prob)
        W_fc2 = cls.__weight_variable([input_shape, num_outputs])
        b_fc2 = cls.__bias_variable([num_outputs])
        return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2), num_outputs

    @classmethod
    def create_graph(cls, input, keep_prob):
        shape = (200, 200, 3)

        # Normalize [0,255] ints to [0,1] floats
        input_float = tf.div(tf.cast(input, tf.float32), 255)

        # Build model
        model = input_float
        model, shape = cls.__conv_layer(shape, model, 5, 5, 3, 8, 2)
        model, shape = cls.__conv_layer(shape, model, 5, 5, 8, 16, 2)
        model, shape = cls.__conv_layer(shape, model, 5, 5, 16, 32, 2)
        model, shape = cls.__conv_layer(shape, model, 5, 5, 32, 64, 5)
        model, shape = cls.__fully_connected_layer(shape, model, 128)
        model, shape = cls.__loss_layer(shape, model, keep_prob, 4)

        return model
