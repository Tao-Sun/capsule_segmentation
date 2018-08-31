import math
import time

import numpy as np
import tensorflow as tf

from python.baseline.unet_hippo import inference
from python.data.hippo import hippo_input
from python.data.affnist import affnist_input
from python.data.pascal import pascal_input
from python.data.caltech import caltech_input
from python.utils import accuracies

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/tmp/hippo',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           """Directory where to read input data """)
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.flags.DEFINE_string('dataset', 'caltech',
                       'The dataset to use for the experiment.'
                       'hippo, affnist, caltech.')
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
tf.app.flags.DEFINE_string('checkpoint_file', None,
                            """checkpoint file to load.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 2,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 576,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('split', 'test',
                            """validation or test, split to evaluate.""")
tf.app.flags.DEFINE_integer('error_block_size', 20,
                            """size of error blocks.""")
tf.app.flags.DEFINE_integer('num_classes', 11,
                            """How many classes to classify.""")


def get_batched_features(batch_size):
    batched_features = None
    if FLAGS.dataset == 'hippo':
        batched_features = hippo_input.inputs(FLAGS.split,
                                              FLAGS.data_dir,
                                              batch_size,
                                              file_start=FLAGS.file_start,
                                              file_end=FLAGS.file_end)
    elif FLAGS.dataset == 'affnist':
        batched_features = affnist_input.inputs(FLAGS.split,
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end,
                                                num_classes=FLAGS.num_classes)
    elif FLAGS.dataset == 'pascal':
        batched_features = pascal_input.inputs(FLAGS.split,
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end,
                                                num_classes=FLAGS.num_classes)
    elif FLAGS.dataset == 'caltech':
        batched_features = caltech_input.inputs(FLAGS.split,
                                                FLAGS.data_dir,
                                                batch_size,
                                                file_start=FLAGS.file_start,
                                                file_end=FLAGS.file_end)

    return batched_features


def eval_once(summary_writer, img_indices_op, images_op, inferred_labels_op, labels_op, summary_op, num_classes):
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
        if FLAGS.checkpoint_file is None:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        else:
            print('checkpoint file to find:' + FLAGS.checkpoint_file)
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, FLAGS.checkpoint_file)
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

            total_dices = [[]] * (num_classes)
            total_error_blocks = [[]] * (num_classes)

            total_true_positives = np.zeros(num_classes)
            total_class_sums = np.zeros(num_classes)
            total_false_positives = np.zeros(num_classes)
            total_accu_stats = np.array([total_true_positives, total_class_sums, total_false_positives])

            if FLAGS.dataset == "hippo":
                assert(FLAGS.subject_size % FLAGS.batch_size == 0)
                group_size = FLAGS.subject_size / FLAGS.batch_size
                print('group size: %d' % group_size)
                prediction_batches = []
                target_batches = []

            step = 0
            group = 0
            while step < num_iter and not coord.should_stop():
                if step % 1 == 0:
                    print("step: %d" % step)

                indices_batch, image_batch, prediction_batch, target_batch = \
                    sess.run([img_indices_op, images_op, inferred_labels_op, labels_op])

                if FLAGS.dataset == "hippo":
                    prediction_batches.append(prediction_batch)
                    target_batches.append(target_batch)
                    if (step + 1) % group_size == 0:
                        group += 1

                        prediction_subject = np.vstack(prediction_batches)
                        target_subject = np.vstack(target_batches)
                        prediction_batches = []
                        target_batches = []

                        subject_dice_0, subject_dice_1 = hippo_input.subject_dice(target_subject, prediction_subject)
                        print("subject_dices: %f, %f" % (subject_dice_0, subject_dice_1))
                        total_dices[0].append(subject_dice_0)

                        hippo_input.save_nii(target_subject, prediction_subject, FLAGS.data_dir, group)
                elif FLAGS.dataset == 'affnist':
                    batch_dices, batch_accu_stats, batch_error_blocks_num = \
                        affnist_input.batch_eval(target_batch, prediction_batch, num_classes, FLAGS.error_block_size)
                    # print(batch_accuracies)
                    # print(batch_dices)
                    # print

                    for i in range(len(target_batch)):
                        if indices_batch[i] < 3000:
                            affnist_input.save_files(FLAGS.data_dir, 'unet', indices_batch[i], image_batch[i],
                                                     target_batch[i], prediction_batch[i], num_classes)

                    for i in range(num_classes):
                        total_dices[i] = np.concatenate((total_dices[i], batch_dices[i]))
                        total_error_blocks[i] = np.concatenate((total_error_blocks[i], batch_error_blocks_num[i]))
                        total_accu_stats += batch_accu_stats
                elif FLAGS.dataset == 'pascal':
                    batch_dices, batch_accu_stats, batch_error_blocks_num = \
                        pascal_input.batch_eval(target_batch, prediction_batch, num_classes,
                                                 FLAGS.error_block_size)
                    # print(batch_accuracies)
                    # print(batch_dices)
                    # print

                    for i in range(len(target_batch)):
                        if indices_batch[i] < 3000:
                            pascal_input.save_files(FLAGS.data_dir, 'unet', indices_batch[i], image_batch[i],
                                                     target_batch[i], prediction_batch[i], num_classes)

                    for i in range(num_classes):
                        total_dices[i] = np.concatenate((total_dices[i], batch_dices[i]))
                        total_error_blocks[i] = np.concatenate((total_error_blocks[i], batch_error_blocks_num[i]))
                        total_accu_stats += batch_accu_stats

                    # batch_dices = caltech_input.batch_eval(indices_batch, target_batch, prediction_batch, num_classes)
                    # # print(str(batch_dice_0))
                    # # print(batch_dice_1)
                    # # print
                    # for i in range(num_classes - 1):
                    #     total_dices[i] = np.concatenate((total_dices[i], batch_dices[i]))

                step += 1

            # a1, b1, c1 = total_accu_stats[0][1:2], total_accu_stats[1][1:2], total_accu_stats[2][1:2]
            # a2, b2, c2 = total_accu_stats[0][3:9], total_accu_stats[1][3:9], total_accu_stats[2][3:9]
            # a3, b3, c3 = [total_accu_stats[0][10]], [total_accu_stats[1][10]], [total_accu_stats[2][10]]
            # print("a1:" + str(a1))
            # print("a2:" + str(a2))
            # print("a3:" + str(a3))
            # global_accuracy, class_accuracies, class_mean_accuracy, mIoU = \
            #     accuracies(np.concatenate((a1, a2, a3)), np.concatenate((b1, b2, b3)), np.concatenate((c1, c2, c3)))

            # global_accuracy, class_accuracies, mIoUs = \
            #     accuracies(total_accu_stats[0], total_accu_stats[1], total_accu_stats[2])
            # print('\nglobal accuracy: %f' % global_accuracy)
            # print('mean accuracy: %f' % np.nanmean(class_accuracies))
            # print('mIoU: %f\n' % np.nanmean(mIoUs))

            global_error_blocks= []
            for i in range(1, num_classes):
                print("class: %d" % i)
                mean_dices, std_dices = np.mean(total_dices[i]), np.std(total_dices[i])
                # mean_block_errors = np.mean(total_error_blocks[i])
                # total_block_errors = np.sum(total_error_blocks[i])
                # global_error_blocks.extend(total_error_blocks[i].tolist())

                # print('class: %d, accuracy: %f, IoU: %f' % (i, class_accuracies[i], mIoUs[i]))
                print('mean dices:  %f' % mean_dices)
                print('dices std: %f' % std_dices)
                # print('total block errors:  %f' % total_block_errors)
                # print('mean block errors:  %f' % mean_block_errors)
                # print('\nmean accuracies:  %f' % mean_accu)

            # print('\ntotal error blocks:  %f' % np.sum(global_error_blocks))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            # summary.value.add(tag='mean_dices_0', simple_value=mean_dices_0)
            # summary.value.add(tag='std_dices_0', simple_value=std_dices_0)
            # summary.value.add(tag='mean_dices_1', simple_value=mean_dices_1)
            # summary.value.add(tag='std_dices_1', simple_value=std_dices_1)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        batched_features = get_batched_features(FLAGS.batch_size)

        img_indices = batched_features['indices']
        images, labels = batched_features['images'], batched_features['pixel_labels']
        num_classes = batched_features['num_classes']

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [img_indices, images, labels], capacity=2 * FLAGS.num_gpus)

        img_indices_op, images_op, labels_op = batch_queue.dequeue()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        label_logits = inference(images_op, num_classes)
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
            eval_once(summary_writer, img_indices_op, images_op, inferred_labels_op, labels_op, summary_op, num_classes)
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
