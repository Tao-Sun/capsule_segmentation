import math
import time

import numpy as np
import tensorflow as tf

from python.capsnet_1d.capsnet_1d import inference
from python.data.hippo.hippo_input import dice
from python.data.hippo.hippo_input import save_nii
from python.data.hippo.hippo_input import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/tmp/hippo',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           """Directory where to read input data """)
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('batch_size', 24,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('subject_size', 48,
                            """How many batches constitute a subject.""")
tf.app.flags.DEFINE_integer('file_start', 1,
                            """Start file no.""")
tf.app.flags.DEFINE_integer('file_end', 110,
                            """End file no.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(summary_writer, inferred_labels_op, labels_op, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=1000)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            print('checkpoint file: %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print(global_step)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            total_sample_count = num_iter * FLAGS.batch_size

            total_dices_0 = []
            total_dices_1 = []
            assert(FLAGS.subject_size % FLAGS.batch_size == 0)
            group_size = FLAGS.subject_size / FLAGS.batch_size
            print('group size: %d' % group_size)
            prediction_batches = []
            target_batches = []

            step = 0
            group = 0
            while step < num_iter and not coord.should_stop():
                prediction_batch, target_batch = sess.run([inferred_labels_op, labels_op])

                prediction_batches.append(prediction_batch)
                target_batches.append(target_batch)
                if (step + 1) % group_size == 0:
                    print('step: %d' % step)
                    group += 1

                    prediction_subject = np.vstack(prediction_batches)
                    target_subject = np.vstack(target_batches)

                    subject_dice_0, subject_dice_1 = dice(target_subject, prediction_subject)
                    print("subject_dices: %f, %f" % (subject_dice_0, subject_dice_1))
                    total_dices_0.append(subject_dice_0)
                    total_dices_1.append(subject_dice_1)

                    save_nii(target_subject, prediction_subject, FLAGS.data_dir, group)

                    prediction_batches = []
                    target_batches = []

                step += 1

            mean_dices_0, std_dices_0 = np.mean(total_dices_0), np.std(total_dices_0)
            mean_dices_1, std_dices_1 = np.mean(total_dices_1), np.std(total_dices_0)
            print('\nmean dices:  %f, %f' % (np.mean(total_dices_0), np.mean(total_dices_1)))
            print('dices std: %f, %f' % (np.std(total_dices_0), np.std(total_dices_0)))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='mean_dices_0', simple_value=mean_dices_0)
            summary.value.add(tag='std_dices_0', simple_value=std_dices_0)
            summary.value.add(tag='mean_dices_1', simple_value=mean_dices_1)
            summary.value.add(tag='std_dices_1', simple_value=std_dices_1)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        batched_features = inputs('test',
                                  FLAGS.data_dir,
                                  FLAGS.batch_size,
                                  file_start=FLAGS.file_start,
                                  file_end=FLAGS.file_end
                                  )

        images, labels = batched_features['images'], batched_features['labels']
        num_classes = batched_features['num_classes']

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * FLAGS.num_gpus)

        image_batch, labels_op = batch_queue.dequeue()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        class_caps_activations, remakes_flatten, label_logits = inference(image_batch, num_classes)
        inferred_labels_op = tf.argmax(label_logits, axis=3)

        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, g)

        while True:
            eval_once(summary_writer, inferred_labels_op, labels_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
