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

    digit_nums = np.zeros(4)
    for i in range(len(images)):
        img = np.reshape(images[i], (40, 40))

        if img is not None:
            # plt.imshow(cropped_img)
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            image = np.array(img, dtype=np.uint8)
            label = int(labels[i])

            if label in digits:
                if label == digits[0]:
                    label_class = 1
                    digit_nums[0] += 1
                elif label == digits[1]:
                    label_class = 2
                    digit_nums[1] += 1
                # elif label == digits[2]:
                #     label_class = 3
                #     digit_nums[2] += 1
                # elif label == digits[3]:
                #     label_class = 4
                #     digit_nums[3] += 1

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

    if split == 'train':
        shift, pad = 6, 6

        fidx = 0  # tfrecords file index
        iidx = 0  # files written into tfrecords
        oidx = 0  # files read from dataset and in the digits
        file_images = []
        file_labels = []
        while oidx < images_num:
            image, label = images[:, oidx], labels[oidx]
            if label in digits:
                image = np.reshape(image, (28, 28))
                padded_image = np.pad(image, pad, 'constant')

                for i in np.arange(-shift, shift + 1):
                    for j in np.arange(-shift, shift + 1):
                        image = _shift_2d(padded_image, (i, j), shift)
                        file_images.append(image)
                        file_labels.append(label)

                        iidx += 1
                        if iidx > FLAGS.file_size:
                            digit_nums = convert_images(np.array(file_images), np.array(file_labels), digits, fidx)
                            print("File example nums: %s" % digit_nums)
                            fidx += 1
                            iidx = 0
                            file_images = []
                            file_labels = []
            oidx += 1
    elif split == 'test' or split == 'validation':
        start = 0
        end = min(FLAGS.file_size, images_num)
        i = 0
        total_nums = np.zeros(4)
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
        '--digits',
        type=list,
        default=[0, 8],
        help='Digits to convert.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
