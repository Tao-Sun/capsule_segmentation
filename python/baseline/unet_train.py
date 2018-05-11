import os.path
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from python.baseline.unet import inference, loss
from python.data.affnist import affnist_input
from python.data.hippo import hippo_input
from python.data.caltech import caltech_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           """Directory where to read input data """)
tf.flags.DEFINE_string('dataset', 'caltech',
                       'The dataset to use for the experiment.'
                       'hippo, affnist, caltech.')
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('file_start', 0,
                            """Start file no.""")
tf.app.flags.DEFINE_integer('file_end', 11,
                            """End file no.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def get_batched_features(batch_size):
    batched_features = None
    if FLAGS.dataset == 'hippo':
        batched_features = hippo_input.inputs('train',
                                              FLAGS.data_dir,
                                              batch_size,
                                              file_start=FLAGS.file_start,
                                              file_end=FLAGS.file_end)
    elif FLAGS.dataset == 'affnist':
        batched_features = affnist_input.inputs('train',
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end)
    elif FLAGS.dataset == 'caltech':
        batched_features = caltech_input.inputs('train',
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end)

    return batched_features


def tower_loss(scope, images, labels2d, num_classes):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    label_logits = inference(images, num_classes)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(labels2d, label_logits, num_classes)

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


def train(hparams):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        learning_rate = tf.train.exponential_decay(
            learning_rate=hparams.learning_rate,
            global_step=global_step,
            decay_steps=hparams.decay_steps,
            decay_rate=hparams.decay_rate)
        learning_rate = tf.maximum(learning_rate, 1e-6)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            tower_grads = []
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        batch_size = FLAGS.batch_size // max(1, FLAGS.num_gpus)
                        batched_features = get_batched_features(batch_size)

                        image_batch = batched_features['images']
                        label_batch = batched_features['pixel_labels']
                        num_classes = batched_features['num_classes']

                        print('shape of image batch:' + str(image_batch.get_shape()))

                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, image_batch, label_batch, num_classes)

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
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # Print stats
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            # format_str = ('\n%s: step %d')
            # print(format_str % (datetime.now(), step))

            start_time = time.time()
            _, loss_value, lr_value = sess.run([train_op, loss, learning_rate])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 50 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch), learning_rate= %.8f')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch, lr_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 300 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.summary_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.9,
        decay_steps=500,
        learning_rate=0.001,
    )


def main(argv=None):  # pylint: disable=unused-argument
    hparams = default_hparams()

    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    train(hparams)


if __name__ == '__main__':
    tf.app.run()
