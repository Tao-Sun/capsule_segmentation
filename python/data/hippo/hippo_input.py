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

import nibabel as nib
import numpy as np
import tensorflow as tf
from python.utils import dice_ratio


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label_1': tf.FixedLenFeature([], tf.int64),
            'label_2': tf.FixedLenFeature([], tf.int64),
            'name': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    height = tf.cast(features['height'], tf.int32) # tf.to_int64(features['height'])
    width = tf.cast(features['width'], tf.int32) #tf.to_int64(features['width'])
    label_class_1 = tf.cast(features['label_1'], tf.int32)
    label_class_2 = tf.cast(features['label_2'], tf.int32)
    name = tf.cast(features['name'], tf.string)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [1, height, width])
    image.set_shape([1, 24, 56])

    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [height, width])
    label.set_shape([24, 56])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(tf.cast(label, tf.float32) * (1. / 255), tf.int32)

    return image, label, label_class_1, label_class_2, name


def inputs(split, data_dir, batch_size, file_start, file_end, num_classes=2):
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
    file_names = None
    if split == 'train':
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in (range(1, 99))]
    elif split == 'test':
        file_names = [os.path.join(data_dir, str(i) + '.tfrecords') for i in range(99, file_end+1)]

    with tf.name_scope('input'):
        shuffle = None
        if split == 'train':
            shuffle = True
        elif split == 'test':
            shuffle = False

        filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label, label_class_1, label_class_2, name = read_and_decode(filename_queue)

        features = {
            'images': image,
            'pixel_labels': label,
            'indices': name
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

        batched_features['label_class'] = tf.one_hot(label_class_1, num_classes) + tf.one_hot(label_class_2, num_classes)
        batched_features['num_classes'] = num_classes

        return batched_features


def subject_dice(target_subject, prediction_subject):
    # print(target_subject.shape)
    # print(prediction_subject.shape)
    # subject_intersection_0 = 0.0
    # subject_union_0 = 0.0
    # subject_intersection_1 = 0.0
    # subject_union_1 = 0.0
    # for i in range(target_subject.shape[0]):
    #     # print("targets[i] shape" + str(targets[i].shape))
    #     # print("predictions[i] shape" + str(predictions[i].shape))
    #     target_0 = target_subject[i].flatten()
    #     prediction_0 = prediction_subject[i].flatten()
    #     # print(target_0[1:100])
    #     # print(prediction_0[1:100])
    #     intersection_0 = np.sum(np.multiply(target_0, prediction_0))
    #     subject_intersection_0 += intersection_0
    #     union_0 = np.sum(target_0) + np.sum(prediction_0)
    #     subject_union_0 += union_0
    #
    #     target_1 = 1 - target_0
    #     # print(target_1[1:100])
    #     prediction_1 = 1 - prediction_0
    #     # print(prediction_1[1:100])
    #     intersection_1 = np.sum(np.multiply(target_1, prediction_1))
    #     subject_intersection_1 += intersection_1
    #     # print("positive_target_indices shape:" + str(positive_target_indices))
    #     # print("positive_pred_indices shape:" + str(positive_pred_indices))
    #
    #     union_1 = np.sum(target_1) + np.sum(prediction_1)
    #     subject_union_1 += union_1
    #
    # smooth= 1.0e-9
    # batch_dice_0 = (2.0 * subject_intersection_0 + smooth) / (subject_union_0 + smooth)
    # batch_dice_1 = (2.0 * subject_intersection_1 + smooth) / (subject_union_1 + smooth)
    _, dice = dice_ratio(target_subject, prediction_subject)
    return dice


def save_nii(target, prediction, save_dir, file_no):
    batch_size = target.shape[0]
    height = target.shape[1]
    width = target.shape[2]

    target_nii = nib.Nifti1Image(np.reshape(target.astype(np.int32), (batch_size, height, width)), np.eye(4))
    prediction_nii = nib.Nifti1Image(np.reshape(prediction.astype(np.int32), (batch_size, height, width)),
                                     np.eye(4))
    nib.save(target_nii, os.path.join(save_dir, 't_' + str(file_no) + '.nii.gz'))
    nib.save(prediction_nii, os.path.join(save_dir, 'p_' + str(file_no) + '.nii.gz'))
