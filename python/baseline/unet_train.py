import os.path
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from model_loader import get_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('model', 'hippo',
                           """Model to load """)
tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           """Directory where to read input data """)
tf.flags.DEFINE_string('dataset', 'caltech',
                       'The dataset to use for the experiment.'
                       'hippo, affnist, caltech.')
tf.flags.DEFINE_string('optimizer', 'sgd',
                       'The optimizer to use for the experiment.'
                       'sgd, adam.')
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('file_start', 1,
                            """Start file no.""")
tf.app.flags.DEFINE_integer('file_end', 110,
                            """End file no.""")
tf.app.flags.DEFINE_integer('max_steps', 600000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """How many classes to classify.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('restore_sess', False,
                            """Whether to restore from saver.""")


def get_batched_features(model, batch_size):
    batched_features = model.data_input.inputs('train',
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end)


    return batched_features


def tower_loss(model, scope, images, labels2d, num_classes):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    label_logits = model.inference(images, num_classes, training=True)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model.loss(labels2d, label_logits, num_classes)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            if g is None: print(v)
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(model):
    """Train CIFAR-10 for a number of steps."""
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    hparams = model.default_hparams()
    with tf.Graph().as_default():
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        if FLAGS.optimizer == 'adam':
            learning_rate = tf.train.exponential_decay(
                learning_rate=hparams.learning_rate,
                global_step=global_step,
                decay_steps=hparams.decay_steps,
                decay_rate=hparams.decay_rate)
            learning_rate = tf.maximum(learning_rate, 1e-8)
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
        elif FLAGS.optimizer == 'sgd':
            learning_rate = tf.constant(hparams.learning_rate)
            # learning_rate = tf.train.inverse_time_decay(
            #     learning_rate=hparams.learning_rate,
            #     global_step=global_step,
            #     decay_steps=hparams.decay_steps,
            #     decay_rate=hparams.decay_rate,
            #     staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate, hparams.momentum)

        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            tower_grads = []
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        batch_size = FLAGS.batch_size // max(1, FLAGS.num_gpus)
                        batched_features = get_batched_features(model, batch_size)

                        image_batch = batched_features['images']
                        label_batch = batched_features['pixel_labels']
                        num_classes = batched_features['num_classes']

                        print('shape of image batch:' + str(image_batch.get_shape()))

                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(model, scope, image_batch, label_batch, num_classes)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = optimizer.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))

        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=100)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        print('total_params: %d\n' % param_stats.total_parameters)

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        if FLAGS.restore_sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.summary_dir)
            assert(ckpt and ckpt.model_checkpoint_path, 'No checkpoint file found')
            # Restores from checkpoint
            print('%s: checkpoint file: %s' % (datetime.now(), ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            restored_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("%s: restored from step: %s" % (datetime.now(), str(restored_step)))
            start_step = int(restored_step) + 1
        else:
            start_step = 0
            sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        for step in range(start_step, FLAGS.max_steps):
            # format_str = ('\n%s: step %d')
            # print(format_str % (datetime.now(), step))

            start_time = time.time()
            if step % 100 != 0:
                sess.run([train_op])
            else:
                _, loss_value, lr_value = sess.run([train_op, loss, learning_rate])
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %e (%.1f examples/sec; %.3f '
                              'sec/batch), learning_rate = %e')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch, lr_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 300 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.summary_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    model = get_model(FLAGS.model)

    if FLAGS.restore_sess is False:
        if tf.gfile.Exists(FLAGS.summary_dir):
            print("Deleting existing summary files!!!\n")
            tf.gfile.DeleteRecursively(FLAGS.summary_dir)
        tf.gfile.MakeDirs(FLAGS.summary_dir)
    train(model)


if __name__ == '__main__':
    tf.app.run()
