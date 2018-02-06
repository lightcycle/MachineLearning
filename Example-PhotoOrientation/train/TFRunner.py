import tensorflow as tf
from tensorflow.python.client import timeline


class TFRunner:
    @classmethod
    def run(cls, op, feed_dict = None, notify_every = 1, restore_checkpoint = None, save_checkpoint = None, profile = None, summary = None, summary_every = 1, batch_result_callback = None):
        # Create Tensorflow session
        sess = tf.Session()

        # Initialize variables defined in the graph so far
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)

        # Restore variable values from a checkpoint file
        var_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'limit_epochs' not in v.name]
        saver = tf.train.Saver(var_list = var_list) if restore_checkpoint or save_checkpoint else None
        cls.__restore_checkpoint(sess, saver, restore_checkpoint)

        # Create a summary writer
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary, sess.graph) if summary else None

        # Start queue runners responsible for loading data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Get profiler options
        run_options, run_metadata = cls.__profile_options(profile)

        # Run graph
        print('Running graph...')
        batch = 1
        try:
            while not coord.should_stop():
                if summary_writer and batch % summary_every == 0:
                    batch_result, batch_summary = sess.run([op, merged_summary], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(batch_summary, batch)
                else:
                    batch_result = sess.run(op, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                cls.__handle_batch_result(batch_result, batch_result_callback)
                cls.__write_terminal_update(batch, notify_every)
                cls.__write_profile(batch, profile, run_metadata)
                batch += 1
        except tf.errors.OutOfRangeError:
            print('Done running graph.')
        finally:
            coord.request_stop()

        # Wait for queue runners to complete
        coord.join(threads)

        # Close summary writer
        if summary_writer:
            summary_writer.close()

        # Save variable values to a checkpoint file
        cls.__save_checkpoint(sess, saver, save_checkpoint)

        # Close Tensorflow session
        sess.close()

    @classmethod
    def __handle_batch_result(cls, batch_result, batch_result_callback):
        if batch_result_callback:
            batch_result_callback(batch_result)

    @classmethod
    def __write_profile(cls, batch, profile, run_metadata):
        if profile and batch == 1:
            with open(profile, 'w') as f:
                tl = timeline.Timeline(run_metadata.step_stats)
                trace = tl.generate_chrome_trace_format()
                f.write(trace)
            print('CPU profiling trace of first batch written to ' + profile)

    @classmethod
    def __write_terminal_update(cls, batch, notify_every):
        if batch % notify_every == 0:
            print('Finished batch ' + str(batch))

    @classmethod
    def __restore_checkpoint(cls, sess, saver, restore_checkpoint):
        if restore_checkpoint is not None:
            saver.restore(sess, restore_checkpoint)
            print('Restoring model from ' + restore_checkpoint)

    @classmethod
    def __save_checkpoint(cls, sess, saver, save_checkpoint):
        if save_checkpoint is not None:
            saver.save(sess, save_checkpoint)
            print('Saved model to ' + save_checkpoint)

    @classmethod
    def __profile_options(cls, profile):
        if profile:
            return tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), tf.RunMetadata()
        else:
            return None, None
