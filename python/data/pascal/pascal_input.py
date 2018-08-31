from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import numpy as np
import skimage.io as io
import tensorflow as tf

from python.utils import dice, accuracy_stats, connected_error_num
from python.data.pascal import LABELS, HEIGHT, WIDTH
import cv2


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'index': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label_classes': tf.VarLenFeature(tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    index = tf.cast(features['index'], tf.int32)
    height = tf.cast(features['height'], tf.int32)  # tf.to_int64(features['height'])
    width = tf.cast(features['width'], tf.int32)  #tf.to_int64(features['width'])
    label_classes = tf.sparse_tensor_to_dense(tf.cast(features['label_classes'], tf.int32))

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    image = tf.transpose(image, perm=[2, 0, 1])
    image.set_shape([3, HEIGHT, WIDTH])

    pixel_labels = tf.decode_raw(features['label_raw'], tf.uint8)
    pixel_labels = tf.reshape(pixel_labels, [height, width])
    pixel_labels.set_shape([HEIGHT, WIDTH])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255)
    image = tf.cast(image, tf.float32)
    mean = tf.constant([104.00698793, 116.66876762, 122.67891434], dtype=tf.float32, shape=[3, 1, 1], name='image_mean')
    image = image - mean
    pixel_labels = tf.cast(pixel_labels, tf.int32)

    return index, image, pixel_labels, label_classes


def inputs(split, data_dir, batch_size, file_start, file_end, num_classes):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """

    # file_num = file_end - file_start + 1
    # test_start_num = int(0.9 * file_num)
    # file_names = None

    # file_names = [os.path.join(data_dir, str(file_idx) + '.tfrecords') for file_idx in range(0, file_end + 1)]
    file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(file_start, file_end + 1)]
    with tf.name_scope('input'):
        shuffle = None
        if split == 'train':
            shuffle = True
        elif split == 'test' and split == 'validation':
            shuffle = False
        filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        index, image, pixel_labels, label_classes = read_and_decode(filename_queue)

        label_class = tf.constant([0.0] * num_classes)
        ite = tf.constant(0)
        cond = lambda i, l: i < tf.shape(label_classes)[0]
        body = lambda i, l: (i + 1, tf.add(l, tf.one_hot(label_classes[i], num_classes)))
        tf.while_loop(cond, body, [ite, label_class])

        features = {
            'indices': index,
            'images': image,
            'pixel_labels': pixel_labels,
            'label_class': label_class
        }

        batched_features = None
        if split == 'train':
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            batched_features = tf.train.shuffle_batch(
                features, batch_size=batch_size, num_threads=2,
                capacity=100 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=batch_size)
        elif split == 'test' or split == 'validation':
            batched_features = tf.train.batch(
                features, batch_size=batch_size,
                num_threads=2,
                capacity=100 + 3 * batch_size)

        batched_features['num_classes'] = num_classes

        return batched_features


def batch_eval(target_batch, prediction_batch, num_classes, error_block_size):
    batch_dices = []
    batch_accuracies = []
    batch_error_blocks_num = []
    for i in range(num_classes):
        batch_dices.append([])
        batch_error_blocks_num.append([])

    batch_true_positives = np.zeros(num_classes)
    batch_class_sums = np.zeros(num_classes)
    batch_mIoU = np.zeros(num_classes)
    batch_accu_stats = np.array([batch_true_positives, batch_class_sums, batch_mIoU])

    for i in range(len(target_batch)):
        for j in range(0, num_classes):
            # print("label 3 number: %d" % len(np.where(target_batch[i].flatten() == 1)[0]))
            if len(np.where(target_batch[i].flatten() == j)[0]) > 0:
                target_label = j
                target_img = np.where(target_batch[i] == target_label, 1, 0)
                prediction_img = np.where(prediction_batch[i] == target_label, 1, 0)
                dice_val = dice(target_img, prediction_img)
                # if j == 3:
                #     print("find:" + str(dice_val))
                batch_accu_stats += \
                    accuracy_stats(target_batch[i].flatten(), prediction_batch[i].flatten(), labels=range(num_classes))

                error_blocks_num = \
                    connected_error_num(target_batch[i], prediction_batch[i], target_label, error_block_size)

                batch_dices[j - 1].append(dice_val)
                batch_error_blocks_num[j - 1].append(error_blocks_num)

    # print(batch_dices)
    # print(np.mean(batch_dices[1]))
    # print(np.mean(batch_dices[0]))

    return batch_dices, batch_accu_stats, batch_error_blocks_num


def save_files(save_dir, model_type, file_no, image, label, prediction, num_classes):
    def make_color_img(grey_img):
        color_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
        for j in range(1, num_classes):
            color_img[np.where(grey_img == j)] = LABELS[j]
        return color_img

    color_label = make_color_img(label)
    color_prediction = make_color_img(prediction)
    color_image = np.array(np.transpose(image, axes=[1, 2, 0]) * 255, dtype=np.uint8)

    image_path = os.path.join(save_dir, model_type, str(file_no) + '_img' + '.png')
    target_path = os.path.join(save_dir, model_type, str(file_no) + '_target' + '.png')
    prediction_path = os.path.join(save_dir, model_type, str(file_no) + '_prediction' + '.png')

    cv2.imwrite(image_path, color_image)
    cv2.imwrite(target_path, color_label)
    cv2.imwrite(prediction_path, color_prediction)