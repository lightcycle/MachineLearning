import tensorflow as tf
import matplotlib.pyplot as plt

temp_f = [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7, 71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5, 76.3]
cricket_chirps_per_s = [20, 16, 19.8, 18.4, 17.1, 15.5, 14.7, 15.7, 15.4, 16.3, 15, 17.2, 16, 17, 14.4]

weight = tf.Variable(0., name ="weight")
bias = tf.Variable(0., name = "bias")


def inference(input):
    with tf.name_scope("Model"):
        return tf.mul(input, weight) + bias


def loss(input, output_expected):
    with tf.name_scope("Loss_Function"):
        output_predicted = inference(input)
        return tf.reduce_sum(tf.squared_difference(output_expected, output_predicted))


def inputs():
    return tf.to_float(temp_f), tf.to_float(cricket_chirps_per_s)


def train(total_loss):
    learning_rate = 0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def show_result_plot():
    plt.plot(temp_f, cricket_chirps_per_s, 'ro', label='Data')
    plt.plot(temp_f, session.run(inference(temp_f)), label='Trained Model')
    plt.show()


with tf.Session() as session:
    tf.initialize_all_variables().run()

    input, output_expected = inputs()

    total_loss_op = loss(input, output_expected)
    train_op = train(total_loss_op)

    writer = tf.train.SummaryWriter("./tensorboard", session.graph)
    total_loss_summary_op = tf.scalar_summary("total_loss", total_loss_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    training_steps = 1000
    for step in range(training_steps):
        _, total_loss_summary = session.run([train_op, total_loss_summary_op])
        writer.add_summary(total_loss_summary, step)

    show_result_plot()

    coord.request_stop()
    coord.join(threads)
    writer.close()
    session.close()