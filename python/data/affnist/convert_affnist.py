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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import scipy.io as sio
import numpy as np
from sets import Set
import random
from python.utils import add_noise
# from matplotlib import pyplot as plt

FLAGS = None
# IIMAGE_SIZE_PX = 28


def _frame(img, max_h, max_w):
    row_range = (np.nonzero(img)[0].min(), np.nonzero(img)[0].max())
    col_range = (np.nonzero(img)[1].min(), np.nonzero(img)[1].max())

    framed_img = None
    if (row_range[1] - row_range[0] <= 28) and (col_range[1] - col_range[0] <= 28):
        framed_img = img[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    return framed_img


def _get_d1_padding(total_padding):
    before = int(random.uniform(0, 1) * total_padding)
    after = total_padding - before
    return before, after


def _crop(img, height, width):
    framed_img = _frame(img, height, width)

    cropped_img = None
    if framed_img is not None:
        framed_shape = framed_img.shape
        padding1_d1 = _get_d1_padding(height - framed_shape[0])
        padding1_d2 = _get_d1_padding(width - framed_shape[1])
        cropped_img = np.pad(framed_img,
                             (padding1_d1, padding1_d2),
                             'constant', constant_values=((0, 0), (0, 0)))

    return cropped_img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _shift_2d(image, shift, max_shift):
    """Shifts the image along each axis by introducing zero.
    Args:
      image: A 2D numpy array to be shifted.
      shift: A tuple indicating the shift along each axis.
      max_shift: The maximum possible shift.
    Returns:
      A 2D numpy array with the same shape of image.
    """
    max_shift += 1
    padded_image = np.pad(image, max_shift, 'constant')
    rolled_image = np.roll(padded_image, shift[0], axis=0)
    rolled_image = np.roll(rolled_image, shift[1], axis=1)
    shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
    return shifted_image


def convert_images(images, labels, digits, fidx):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.dest, str(fidx) + '.tfrecords')
    print('Writing: %s, images num: %d' % (filename, len(images)))
    writer = tf.python_io.TFRecordWriter(filename)

    digit_nums = np.zeros(len(digits))
    for i in range(len(images)):
        img = np.reshape(images[i], (40, 40))

        if img is not None:
            # plt.imshow(cropped_img)
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            image = np.array(img, dtype=np.uint8)
            label = int(labels[i])

            if label in digits:
                index = digits.index(label)
                # Class label of the number 0 is 1, class label of the number 1 is 2, and so on.
                label_class = index + 1
                digit_nums[index] += 1

                max_noise_val = 5
                image = add_noise(image, 1, max_noise_val)
                image_raw = image.tostring()
                label_raw = np.array(np.where(image > 2 * max_noise_val, label_class, 0), dtype=np.uint8).tostring()

                features = tf.train.Features(feature={
                    'index': _int64_feature(i),
                    'height': _int64_feature(image.shape[0]),
                    'width': _int64_feature(image.shape[1]),
                    'label_class': _int64_feature(label_class),
                    'image_raw': _bytes_feature(image_raw),
                    'label_raw': _bytes_feature(label_raw)})

                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
    writer.close()

    return digit_nums


def main(unused_argv):
    data_dir, split = FLAGS.data_dir, FLAGS.split
    data = sio.loadmat(os.path.join(data_dir, split + '.mat'))
    digits = FLAGS.digits
    print('digits: %s' % str(digits))

    images = data['affNISTdata'][0][0][2]
    print('images shape: %s' % str(images.shape))
    labels = data['affNISTdata'][0][0][5][0]
    print('labels shape: %s' % str(labels.shape))
    images_num = images.shape[1]

    start = 0
    end = min(FLAGS.file_size, images_num)
    i = FLAGS.file_start_idx
    total_nums = np.zeros(len(digits))
    while True:
        digit_nums = convert_images(np.transpose(images[:, start:end]), np.transpose(labels[start:end]), digits, i)
        print("File example nums: %s" % digit_nums)
        total_nums = total_nums + digit_nums
        if end < images_num:
            start = end
            end = min(end + FLAGS.file_size, images_num)
            i += 1
        else:
            break

    print("Total example nums: %s" % total_nums)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory to download data files.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        help='train, test or validation.'
    )
    parser.add_argument(
        '--dest',
        type=str,
        default='/tmp/data',
        help='Destination directory.'
    )
    parser.add_argument(
        '--file_size',
        type=int,
        help='The volume of each generated file.'
    )
    parser.add_argument(
        '--file_start_idx',
        type=int,
        default=0,
        help='Start index of written tfrecords files.'
    )
    parser.add_argument(
        '--digits',
        type=list,
        default=range(0, 10),
        help='Digits to convert.'
    )


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
