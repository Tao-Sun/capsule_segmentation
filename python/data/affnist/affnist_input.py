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

from python.utils import dice, accuracy


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    height = tf.cast(features['height'], tf.int32) # tf.to_int64(features['height'])
    width = tf.cast(features['width'], tf.int32) #tf.to_int64(features['width'])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [1, height, width])
    image.set_shape([1, 28, 28])

    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [height, width])
    label.set_shape([28, 28])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(label, tf.int32)

    return image, label


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
    # test_start_num = int(1.0 * file_num)
    file_names = None

    file_names = [os.path.join(data_dir, str(file_idx) + '.tfrecords') for file_idx in range(0, file_end + 1)]
    # if split == 'train':
    #     file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(1, test_start_num)]
    # elif split == 'test':
    #     file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(test_start_num, file_end + 1)]

    with tf.name_scope('input'):
        shuffle = None
        if split == 'train':
            shuffle = True
        elif split == 'test':
            shuffle = False
        filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        features = {
            'images': image,
            'labels': label,
        }

        batched_features = None
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

        batched_features['num_classes'] = 3

        return batched_features


def batch_eval(target_batch, prediction_batch):
    batch_dices_1 = []
    batch_dices_2 = []
    batch_accuracy_1 = []
    batch_accuracy_2 = []

    for i in range(len(target_batch)):
        # print("label 3 number: %d" % len(np.where(target_batch[i].flatten() == 1)[0]))
        if len(np.where(target_batch[i].flatten() == 1)[0]) > 0:
            target_1_img = np.where(target_batch[i] == 1, 1, 0)
            prediction_1_img = np.where(prediction_batch[i] == 1, 1, 0)
            dice_1 = dice(target_1_img, prediction_1_img)
            accuracy_1 = accuracy(target_1_img, prediction_1_img)
            batch_dices_1.append(dice_1)
            batch_accuracy_1.append(accuracy_1)

        # print("label 5 number: %d" % len(np.where(target_batch[i].flatten() == 2)[0]))
        if len(np.where(target_batch[i].flatten() == 2)[0]) > 0:
            target_2_img = np.where(target_batch[i] == 2, 1, 0)
            prediction_2_img = np.where(prediction_batch[i] == 2, 1, 0)
            dice_2 = dice(target_2_img, prediction_2_img)
            accuracy_2 = accuracy(target_2_img, prediction_2_img)
            batch_dices_2.append(dice_2)
            batch_accuracy_2.append(accuracy_2)

        def display_label(label):
            label[np.where(label == 1)] = 50
            label[np.where(label == 2)] = 255
            return label
        cv2.imwrite('target' + str(i) + '.png', display_label(target_batch[i]))
        cv2.imwrite('prediction' + str(i) + '.png', display_label(prediction_batch[i]))

    return batch_dices_1, batch_dices_2, batch_accuracy_1, batch_accuracy_2


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