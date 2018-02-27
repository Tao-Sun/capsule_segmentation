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
import re
import scipy.misc
import skimage.io as io
import numpy as np

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to():
    """Converts a dataset to tfrecords."""
    data_dir = FLAGS.dir
    print('Reading', data_dir)

    images = []
    labels = []
    names = []
    print(FLAGS.start - FLAGS.end + 1)
    j = 0
    for i in range(FLAGS.end - FLAGS.start + 1):
        index = i + 1
        image = io.imread(os.path.join(data_dir, 'hippo' + str(index) + '.png'))
        label = io.imread(os.path.join(data_dir, 'label' + str(index) + '.png'))
        images.append(image)
        labels.append(label)
        names.append('hippo' + str(i))

        if index % FLAGS.vol == 0:
            j += 1
            name = str(int(index / FLAGS.vol))
            _write_file(name, images, labels, names)

            images = []
            labels = []
            names = []
    print(str(j) + ' files written!')


def _write_file(name, images, labels, names):
    filename = os.path.join(FLAGS.des, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(images)):
        image_raw = images[i].tostring()
        label_raw = labels[i].tostring()
        name = names[i]
        features = tf.train.Features(feature={
            'height': _int64_feature(images[i].shape[0]),
            'width': _int64_feature(images[i].shape[1]),
            'name': _bytes_feature(name),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    print(str(len(images)) + ' samples written!')


def main(unused_argv):
    # Get the data.
    convert_to()
    # Convert to Examples and write the result to TFRecords.
    # convert_to(images, labels, FLAGS.name)
    # convert_to(data_sets.validation, 'validation')
    # convert_to(data_sets.test, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        default='/tmp/data',
        help='Directory to download data files.'
    )
    parser.add_argument(
        '--des',
        type=str,
        default='/tmp/data',
        help='Destination directory.'
    )
    parser.add_argument(
        '--start',
        type=int,
        default='1',
        help='Start number of the images and labels.'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='End number of the images and labels.'
    )
    parser.add_argument(
        '--vol',
        type=int,
        help='The volume of each generated file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
