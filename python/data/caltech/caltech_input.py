# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import numpy as np
import skimage.io as io
import tensorflow as tf
import cv2

from python.utils import dice, accuracy_stats

HEIGHT, WIDTH = 28, 28
NUM_CLASSES = 3

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
            'label_1': tf.FixedLenFeature([], tf.int64),
            'label_2': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
            'image_raw_1': tf.FixedLenFeature([], tf.string),
            'label_raw_1': tf.FixedLenFeature([], tf.string),
            'image_raw_2': tf.FixedLenFeature([], tf.string),
            'label_raw_2': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    index = tf.cast(features['index'], tf.int32)
    height = tf.cast(features['height'], tf.int32)  # tf.to_int64(features['height'])
    width = tf.cast(features['width'], tf.int32)  #tf.to_int64(features['width'])
    label_class_1 = tf.cast(features['label_1'], tf.int32)
    label_class_2 = tf.cast(features['label_2'], tf.int32)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [1, height, width])
    image.set_shape([1, HEIGHT, WIDTH])

    pixel_labels = tf.decode_raw(features['label_raw'], tf.uint8)
    pixel_labels = tf.reshape(pixel_labels, [height, width])
    pixel_labels.set_shape([HEIGHT, WIDTH])

    image_1 = tf.decode_raw(features['image_raw_1'], tf.uint8)
    image_1 = tf.reshape(image_1, [1, height, width])
    image_1.set_shape([1, HEIGHT, WIDTH])

    pixel_labels_1 = tf.decode_raw(features['label_raw_1'], tf.uint8)
    pixel_labels_1 = tf.reshape(pixel_labels_1, [height, width])
    pixel_labels_1.set_shape([HEIGHT, WIDTH])

    image_2 = tf.decode_raw(features['image_raw_2'], tf.uint8)
    image_2 = tf.reshape(image_2, [1, height, width])
    image_2.set_shape([1, HEIGHT, WIDTH])

    pixel_labels_2 = tf.decode_raw(features['label_raw_2'], tf.uint8)
    pixel_labels_2 = tf.reshape(pixel_labels_2, [height, width])
    pixel_labels_2.set_shape([HEIGHT, WIDTH])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    features = {}
    features['indices'] = index
    features['images'] = tf.cast(image, tf.float32) * (1. / 255)
    features['pixel_labels'] = tf.cast(pixel_labels, tf.int32)
    features['label_class'] = tf.one_hot(label_class_1, NUM_CLASSES) + tf.one_hot(label_class_2, NUM_CLASSES)
    return features


def inputs(split, data_dir, batch_size, file_start, file_end):
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

    file_num = file_end - file_start + 1
    test_start_num = int(0.8 * file_num)
    file_names = None

    # file_names = [os.path.join(data_dir, str(file_idx) + '.tfrecords') for file_idx in range(0, file_end + 1)]
    if split == 'train':
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(1, test_start_num)]
    elif split == 'test':
        print('test start num: %d' % test_start_num)
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(test_start_num, file_end + 1)]

    with tf.name_scope('input'):
        shuffle = None
        if split == 'train':
            shuffle = True
        elif split == 'test':
            shuffle = False
        filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        features = read_and_decode(filename_queue)
        if split == 'train':
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            batched_features = tf.train.shuffle_batch(
                features, batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        elif split == 'test':
            batched_features = tf.train.batch(
                features, batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size)

        batched_features['num_classes'] = NUM_CLASSES
        return batched_features


def batch_eval(indices_batch, target_batch, prediction_batch, num_classes):
    batch_dices = []
    # batch_accuracies = []
    for i in range(num_classes - 1):
        batch_dices.append([])
        # batch_accuracies.append([])

    for i in range(len(target_batch)):
        for j in range(1, num_classes):
            # print("label 3 number: %d" % len(np.where(target_batch[i].flatten() == 1)[0]))
            target_img = np.where(target_batch[i] == j, 1, 0)
            prediction_img = np.where(prediction_batch[i] == j, 1, 0)
            dice_val = dice(target_img, prediction_img)
            # accu_val = accuracy(target_img, prediction_img)

            batch_dices[j - 1].append(dice_val)
                # batch_accuracies[j - 1].append(accu_val)

        def display_target(label):
            for j in range(1, num_classes):
                label[np.where(label == j)] = j * 255.0 / (num_classes - 1)
            return label

        def display_label(label):
            class_colors = [[0, 255, 0], [0, 0, 255]]
            color_label = np.zeros((label.shape[0], label.shape[1], 3))
            for j in range(1, num_classes):
                color_label[np.where(label == j)] = class_colors[j % 2]

            return color_label

        cv2.imwrite('target' + str(indices_batch[i]) + '.png', display_target(target_batch[i]))
        cv2.imwrite('prediction' + str(indices_batch[i]) + '.png', display_label(prediction_batch[i]))

    return batch_dices  # , batch_accuracies


if __name__ == '__main__':
    tfrecords_filename = os.path.join(sys.argv[1], '1.tfrecords')
    print(tfrecords_filename)

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)
    resized_image, resized_annotation = read_and_decode(filename_queue)

    images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        def frame(img):
            row_range = (np.nonzero(img)[0].min(), np.nonzero(img)[0].max())
            col_range = (np.nonzero(img)[1].min(), np.nonzero(img)[1].max())

            zone = (row_range, col_range)
            return zone

        img, anno = sess.run([images, annotations])

        # Let's read off 3 batches just for example
        for i in range(1):
            img, anno = sess.run([images, annotations])
            print('images shape: %s' % str(img.shape))
            print('anno shape: %s' % str(anno.shape))

            # We selected the batch size of two
            # So we should get two image pairs in each batch
            # Let's make sure it is random
            img1 = img[0, 0, :, :]
            # frame1 = frame(img)
            anno1 = anno[0, :, :]
            frame2 = frame(anno1)

            framed1 = np.where(img1[frame2[0][0]:frame2[0][1], frame2[1][0]:frame2[1][1]]>0, 1, 0)
            print("image example:")
            print(framed1.shape)
            io.imshow(img1, cmap='gray')
            io.show()

            framed2 = np.where(anno1[frame2[0][0]:frame2[0][1], frame2[1][0]:frame2[1][1]]>0, 1, 0)
            print("anno example:")
            print(framed2.shape)
            diff = np.subtract(framed1, framed2)
            print('diff:')
            print(diff)
            io.imshow(anno1, cmap='gray')
            io.show()

            io.imshow(img[1, 0, :, :], cmap='gray')
            io.show()

            io.imshow(anno[1, :, :], cmap='gray')
            io.show()

        coord.request_stop()
        coord.join(threads)