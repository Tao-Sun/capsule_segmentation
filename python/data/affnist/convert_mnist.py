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
from python.utils import add_noise

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




def convert(images, labels, index, digits, occlusion):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.dest, str(index) + '.tfrecords')
    print('Writing: %s, images num: %d' % (filename, len(images)))
    writer = tf.python_io.TFRecordWriter(filename)

    digit_nums = np.zeros(len(digits))
    for i in range(len(images)):
        img = np.reshape(images[i], (28, 28))

        if img is not None:
            # plt.imshow(cropped_img)
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            image = np.array(img, dtype=np.uint8)
            label = int(labels[i])

            if label in digits:
                index = digits.index(label)
                label_class = index + 1
                digit_nums[index] += 1

                if occlusion & (index > 44):
                    image[12:16, :] = 0
                #cv2.imwrite(str(i) + "_.png", _add_noise(image, 1, max_noise_val))
                max_noise_val = 5
                image = add_noise(image, 1, max_noise_val)
                image_raw = image.tostring()
                label_img = np.array(np.where(image > 2 * max_noise_val, label_class, 0), dtype=np.uint8)
                label_raw = label_img.tostring()

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
    digits = FLAGS.digits
    occlusion = True if FLAGS.occlusion == 'True' else False
    print('digits: %s' % str(digits))
    print('occlusion: %s' % str(occlusion))

    data = sio.loadmat(os.path.join(data_dir, split + '.mat'))

    images = data['affNISTdata'][0][0][2]
    labels = data['affNISTdata'][0][0][5][0]
    images_num = images.shape[1]

    print(images_num)
    print(images.shape)
    print(labels.shape)
    start = 0
    end = min(FLAGS.file_size, images_num)
    i = 0
    total_nums = np.zeros(len(digits))
    while True:
        digit_nums = convert(np.transpose(images[:, start:end]), np.transpose(labels[start:end]), i, digits, occlusion)
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
        default='train',
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
        default=[0, 1, 2, 3, 4],
        help='Digits to convert.'
    )
    parser.add_argument(
        '--occlusion',
        type=str,
        default='False',
        help='Whether to generate occlusion parts for digits.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
