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
import os, glob
import sys

import tensorflow as tf
import cv2
import scipy.io as sio
import numpy as np
from python.data.pascal import LABELS, HEIGHT, WIDTH

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _generate_mask(file_name):
    color_mask = cv2.imread(file_name, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(color_mask)
    color_mask = cv2.merge([r, g, b])

    mask = np.zeros((color_mask.shape[0], color_mask.shape[1]), dtype=np.uint8)
    for i, label in enumerate(LABELS[1:11]):
        mask[np.where(np.all(color_mask == label, axis=-1))] = i + 1
    return mask


def convert(images, labels, tf_index):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.dest, str(tf_index) + '.tfrecords')
    print('Writing: %s, images num: %d' % (filename, len(images)))
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(len(images)):
        image_raw = images[i].tostring()
        label_raw = labels[i].tostring()
        label_classes = np.unique(labels[i])

        features = tf.train.Features(feature={
            'index': _int64_feature(i),
            'height': _int64_feature(images[i].shape[0]),
            'width': _int64_feature(images[i].shape[1]),
            'label_classes': _int64_features(label_classes),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def main(unused_argv):
    data_dir = FLAGS.data_dir
    anno_dir = FLAGS.anno_dir
    os.chdir(anno_dir)

    i = 0
    total_nums = 0
    images, labels = [], []
    tf_index = 0
    label_dist = np.zeros(10)

    for file_name in glob.glob('*.png'):
        file_suffix = file_name[:-4]
        image_path = os.path.join(data_dir, file_suffix + '.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_shape = image.shape[0:2]

        # if min(img_shape) <= min((HEIGHT, WIDTH)):
        label = _generate_mask(file_name)
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:
            for label_index in unique_labels:
                if label_index != 0:
                    # print(label_index)
                    label_dist[label_index-1] += 1
                    # print(label_dist)

            # if img_shape[0] < img_shape[1]:
            #     image = np.transpose(image, axes=[1, 0, 2])
            #     label = np.transpose(label)
            #     img_shape = image.shape[0:2]

            if img_shape[0] < HEIGHT or img_shape[1] < WIDTH:
                h_diff = HEIGHT - img_shape[0]
                padded_h = int(h_diff / 2), h_diff - int(h_diff / 2)
                w_diff = WIDTH - img_shape[1]
                padded_w = int(w_diff / 2), w_diff - int(w_diff / 2)

                image = np.pad(image, (padded_h, padded_w, (0, 0)),
                               'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                label = np.pad(label, (padded_h, padded_w),
                               'constant', constant_values=((0, 0), (0, 0)))

            assert (image.shape == (HEIGHT, WIDTH, 3))
            assert (image.shape[0:2] == label.shape)

            images.append(image)
            labels.append(label)

            i += 1
            if len(images) == FLAGS.file_size:
                convert(images, labels, tf_index)
                total_nums += len(images)

                images, labels = [], []
                tf_index += 1

    if len(images) > 0:
        convert(images, labels, tf_index)
        total_nums += len(images)

    print("Total example nums: %s" % i)
    print("Total converted example nums: %s" % total_nums)
    print("label distribution: %s" % str(label_dist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory to download data files.'
    )
    parser.add_argument(
        '--anno_dir',
        type=str,
        default='/tmp/data',
        help='Directory to download labels files.'
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
