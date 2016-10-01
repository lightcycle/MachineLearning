import tensorflow as tf

# Placeholders for providing data to graph
with tf.name_scope('inputs') as scope:
    X = tf.placeholder(tf.float32,
                       name = "temperature")
    Y = tf.placeholder(tf.float32,
                       name = "chirp_freq")

# Model (Linear)
with tf.name_scope('model') as scope:
    weight = tf.Variable(0., name = "weight")
    bias = tf.Variable(0., name = "bias")
    modeled_Y = tf.add(tf.mul(X, weight), bias)

# Loss Function (Mean Squared Error)
with tf.name_scope('loss_function') as scope:
    loss = tf.reduce_mean(
        tf.squared_difference(Y, modeled_Y))

with tf.Session() as session:
    summary_writer = tf.train.SummaryWriter("linear_fitting_graph_tensorboard", session.graph)
    summary_writer.close()
