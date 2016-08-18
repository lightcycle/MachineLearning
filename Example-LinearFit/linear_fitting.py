import tensorflow as tf
import matplotlib.pyplot as plt

temp_f = [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7, 71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5, 76.3]
cricket_chirps_per_s = [20, 16, 19.8, 18.4, 17.1, 15.5, 14.7, 15.7, 15.4, 16.3, 15, 17.2, 16, 17, 14.4]

# Placeholders for providing data to computation graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Linear Model
weight = tf.Variable(0., name = "weight")
bias = tf.Variable(0., name = "bias")
modeled_Y = tf.add(tf.mul(X, weight), bias)

# Loss Function
loss = tf.reduce_sum(tf.squared_difference(Y, modeled_Y))

# Training Operation
learning_rate = 0.000001
training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as session:
    tf.initialize_all_variables().run()

    writer = tf.train.SummaryWriter("./tensorboard", session.graph)
    loss_summary_op = tf.scalar_summary("total_loss", loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    training_steps = 25
    for step in range(training_steps):
        _, total_loss_summary = session.run([training_op, loss_summary_op], feed_dict={X: temp_f, Y: cricket_chirps_per_s})
        writer.add_summary(total_loss_summary, step)

    plt.plot(temp_f, cricket_chirps_per_s, 'ro', label='Data')
    plt.plot(temp_f, session.run(modeled_Y, feed_dict={X: temp_f}), label='Trained Model')
    plt.show()

    coord.request_stop()
    coord.join(threads)
    writer.close()
    session.close()