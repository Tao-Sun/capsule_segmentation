# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from python.capsnet_1d.capsnet_1d import inference, loss
from python.data.hippo.hippo_input import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           """Directory where to read input data """)
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """Number of classes.""")
tf.app.flags.DEFINE_integer('batch_size', 12,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('file_start', 1,
                            """Start file no.""")
tf.app.flags.DEFINE_integer('file_end', 110,
                            """End file no.""")

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)


tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')


def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=500,
        learning_rate=0.001,
    )


def task_loss(images, labels2d, num_classes):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    class_caps_activations, remakes_flatten, label_logits = inference(images, num_classes)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    loss(images, labels2d, class_caps_activations, remakes_flatten, label_logits, num_classes)


def train(name_scope, target, cluster_spec, hparams):
    """Train Inception on a dataset for a number of steps."""
    # Number of workers and parameter servers are inferred from the workers and ps
    # hosts string.
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])
    # If no value is given, num_replicas_to_aggregate defaults to be the number of
    # workers.
    if FLAGS.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

    # Both should be greater than 0 in a distributed training.
    assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                           'num_parameter_servers'
                                                           ' must be > 0.')

    # Choose worker 0 as the chief. Note that any worker could be the chief
    # but there should be only one chief.
    is_chief = (FLAGS.task_id == 0)

    # Ops are assigned to worker by default.
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        # Variables and its related init/assign ops are assigned to ps.
        with tf.contrib.framework.arg_scope(
                [tf.contrib.slim.variable, tf.contrib.slim.get_or_create_global_step()],
                device=tf.contrib.framework.VariableDeviceChooser(num_parameter_servers)):
            # Create a variable to count the number of train() calls. This equals the
            # number of updates applied to the variables.
            global_step = tf.contrib.slim.get_or_create_global_step()

            lr = tf.train.exponential_decay(
                learning_rate=hparams.learning_rate,
                global_step=global_step,
                decay_steps=hparams.decay_steps,
                decay_rate=hparams.decay_rate)
            lr = tf.maximum(lr, 1e-6)

            # Add a summary to track the learning rate.
            tf.summary.scalar('learning_rate', lr)

            opt = tf.train.AdamOptimizer(lr)

            batched_features = inputs('train',
                                      FLAGS.data_dir,
                                      FLAGS.batch_size,
                                      file_start=FLAGS.file_start,
                                      file_end=FLAGS.file_end
                                      )

            image_batch = batched_features['images']
            label_batch = batched_features['labels']
            num_classes = batched_features['num_classes']

            print('shape of image batch:' + str(image_batch.get_shape()))

            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            task_loss(image_batch, label_batch, num_classes)

            # Assemble all of the losses for the current tower only.
            losses = tf.get_collection('losses', name_scope)

            # Calculate the total loss for the current tower.
            total_loss = tf.add_n(losses, name='total_loss')

            if is_chief:
                # Compute the moving average of all individual losses and the
                # total loss.
                # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                # loss_averages_op = loss_averages.apply(losses + [total_loss])

                # Attach a scalar summmary to all individual losses and the total loss;
                # do the same for the averaged version of the losses.
                for l in losses + [total_loss]:
                    loss_name = l.op.name
                    # Name each loss as '(raw)' and name the moving average version of the
                    # loss as the original loss name.
                    tf.summary.scalar(loss_name + ' (raw)', l)
                    # tf.summary.scalar(loss_name, loss_averages.average(l))

                # Add dependency to compute loss_averages.
                # with tf.control_dependencies([loss_averages_op]):
                #     total_loss = tf.identity(total_loss)

            # Track the moving averages of all trainable variables.
            # Note that we maintain a 'double-average' of the BatchNormalization
            # global statistics.
            # This is not needed when the number of replicas are small but important
            # for synchronous distributed training with tens of workers/replicas.
            # exp_moving_averager = tf.train.ExponentialMovingAverage(
            #     inception.MOVING_AVERAGE_DECAY, global_step)
            #
            # variables_to_average = (
            #     tf.trainable_variables() + tf.moving_average_variables())
            #
            # Add histograms for model variables.
            # for var in variables_to_average:
            #     tf.summary.histogram(var.op.name, var)

            # Create synchronous replica optimizer.
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_replicas_to_aggregate,
                total_num_replicas=num_workers)

            # batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
            # assert batchnorm_updates, 'Batchnorm updates are missing'
            # batchnorm_updates_op = tf.group(*batchnorm_updates)
            # Add dependency to compute batchnorm_updates.
            # with tf.control_dependencies([batchnorm_updates_op]):
            #     total_loss = tf.identity(total_loss)

            # Compute gradients with respect to the loss.
            grads = opt.compute_gradients(total_loss)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')

            # Get chief queue_runners and init_tokens, which is used to synchronize
            # replicas. More details can be found in SyncReplicasOptimizer.
            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()

            # Create a saver.
            saver = tf.train.Saver()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()

            # Build an initialization operation to run below.
            init_op = tf.global_variables_initializer()

            # We run the summaries in the same thread as the training operations by
            # passing in None for summary_op to avoid a summary_thread being started.
            # Running summaries and training operations in parallel could run out of
            # GPU memory.
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=init_op,
                                     summary_op=None,
                                     global_step=global_step,
                                     saver=saver,
                                     save_model_secs=FLAGS.save_interval_secs)

            tf.logging.info('%s Supervisor' % datetime.now())

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)

            # Get a session.
            sess = sv.prepare_or_wait_for_session(target, config=sess_config)

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
            tf.logging.info('Started %d queues for processing input data.',
                            len(queue_runners))

            if is_chief:
                sv.start_queue_runners(sess, chief_queue_runners)
                sess.run(init_tokens_op)

            # Train, checking for Nans. Concurrently run the summary operation at a
            # specified interval. Note that the summary_op and train_op never run
            # simultaneously in order to prevent running out of GPU memory.
            next_summary_time = time.time() + FLAGS.save_summaries_secs
            while not sv.should_stop():
                try:
                    start_time = time.time()
                    loss_value, step = sess.run([train_op, global_step])
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if step > FLAGS.max_steps:
                        break
                    duration = time.time() - start_time

                    if step % 30 == 0:
                        examples_per_sec = FLAGS.batch_size / float(duration)
                        format_str = ('Worker %d: %s: step %d, loss = %.2f'
                                      '(%.1f examples/sec; %.3f  sec/batch)')
                        tf.logging.info(format_str %
                                        (FLAGS.task_id, datetime.now(), step, loss_value,
                                         examples_per_sec, duration))

                    # Determine if the summary_op should be run on the chief worker.
                    if is_chief and next_summary_time < time.time():
                        tf.logging.info('Running Summary operation on the chief.')
                        summary_str = sess.run(summary_op)
                        sv.summary_computed(sess, summary_str)
                        tf.logging.info('Finished running Summary operation.')

                        # Determine the next time for running the summary.
                        next_summary_time += FLAGS.save_summaries_secs
                except:
                    if is_chief:
                        tf.logging.info('Chief got exception while running!')
                    raise

            # Stop the supervisor.  This also waits for service threads to finish.
            sv.stop()

            # Save after the training ends.
            if is_chief:
                saver.save(sess,
                           os.path.join(FLAGS.train_dir, 'model.ckpt'),
                           global_step=global_step)


def main(unused_args):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    hparams = default_hparams()

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
         'worker': worker_hosts},
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id,
        protocol=FLAGS.protocol)

    if FLAGS.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        # Only the chief checks for or creates train_dir.
        if FLAGS.task_id == 0:
            if tf.gfile.Exists(FLAGS.summary_dir):
                tf.gfile.DeleteRecursively(FLAGS.summary_dir)
            tf.gfile.MakeDirs(FLAGS.summary_dir)

        with tf.name_scope('tower_%d' % FLAGS.task_id) as scope:
            train(scope, server.target, cluster_spec, hparams)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
