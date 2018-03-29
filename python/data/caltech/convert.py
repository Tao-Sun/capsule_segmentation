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
import random

import cv2
import matplotlib.patches as patches
import numpy as np
import scipy.io as sio
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path

FLAGS = None

def frame(img):
    row_range = [np.nonzero(img)[0].min(), np.nonzero(img)[0].max()]
    col_range = [np.nonzero(img)[1].min(), np.nonzero(img)[1].max()]

    zone = (row_range, col_range)

    framed_img = img[row_range[0]:(row_range[1] + 1), col_range[0]:(col_range[1] + 1)]
    return framed_img


def pad(framed_img, h, w, coeff1, coeff2, label_class=None):
    framed_h, framed_w = framed_img.shape
    total_padding_h = h - framed_h
    total_padding_w = w - framed_w

    padding_h = (int(coeff1 * total_padding_h), total_padding_h - int(coeff1 * total_padding_h))
    padding_w = (int(coeff2 * total_padding_w), total_padding_w - int(coeff2 * total_padding_w))

    padded_img = np.pad(framed_img,
                        (padding_h, padding_w),
                        'constant', constant_values=((0, 0), (0, 0)))

    if label_class is not None:
        padded_label = np.array(np.where(padded_img > 0, label_class, 0), dtype=np.uint8)
        return padded_img, padded_label
    else:
        return padded_img


def gen_img(file_path, height, width):
    def get_vertices(contour):
        vertices = []
        codes = [Path.MOVETO]
        for i in range(contour.shape[1]):
            vertices.append([contour[0, i], contour[1, i]])
            if i > 0:
                codes.append(Path.LINETO)
        vertices = np.array(vertices)
        return vertices, codes

    def gen_patch():
        contour = sio.loadmat(file_path)['obj_contour']
        row1 = contour[0] / (max(contour[0]) / (height * 0.8))
        row2 = contour[1] / (max(contour[1]) / (width * 0.8))
        scaled_contour = np.vstack((row1, row2))
        vertices, codes = get_vertices(scaled_contour)
        path = Path(vertices, codes)
        return patches.PathPatch(path, facecolor='orange', lw=0)

    def gen_matplot_fig(patch):
        figure = Figure(figsize=(1, 1), dpi=28)
        ax = figure.gca()
        ax.add_patch(patch)
        figure.patch.set_facecolor('black')

        ax.set_xlim(0, height)
        ax.set_ylim(0, width)
        ax.invert_yaxis()
        ax.axis('off')

        canvas = FigureCanvas(figure)
        canvas.draw()
        return figure

    def gen_gray_img(matplot_fig):
        w, h = matplot_fig.canvas.get_width_height()
        buf = np.fromstring(matplot_fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        gray_buf = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY)
        return gray_buf

    patch = gen_patch()
    fig = gen_matplot_fig(patch)
    img = gen_gray_img(fig)
    return img


def merge_images(file_path_1, file_path_2, height, width):
    img1 = gen_img(file_path_1, height, width)
    framed_img_1 = frame(img1)
    h1, w1 = framed_img_1.shape

    img2 = gen_img(file_path_2, height, width)
    framed_img_2 = frame(img2)
    h2, w2 = framed_img_2.shape

    overlap_shape = (min(h1 + h2 - 1, height), min(w1 + w2 - 1, width))

    overlap_h_coeffs = [random.uniform(0, 0.3), random.uniform(0.8, 1.0)]
    random.shuffle(overlap_h_coeffs)
    overlap_w_coeffs = [random.uniform(0, 0.1), random.uniform(0.9, 1.0)]
    random.shuffle(overlap_w_coeffs)
    overlap_img1, overlap_label1 = pad(framed_img_1, overlap_shape[0],
                                       overlap_shape[1], overlap_h_coeffs[0],
                                       overlap_w_coeffs[0], 1)
    overlap_img2, overlap_label2 = pad(framed_img_2, overlap_shape[0],
                                       overlap_shape[1], overlap_h_coeffs[1],
                                       overlap_w_coeffs[1], 2)


    coeff1, coeff2 = random.uniform(0, 1.0), random.uniform(0, 1.0)
    merged_img_1 = pad(overlap_img1, height, width, coeff1, coeff2).astype(np.uint8)
    merged_label_1 = pad(overlap_label1, height, width, coeff1, coeff2).astype(np.uint8)
    merged_img_2 = pad(overlap_img2, height, width, coeff1, coeff2).astype(np.uint8)
    merged_label_2 = pad(overlap_label2, height, width, coeff1, coeff2).astype(np.uint8)

    shuffle = [0, 1]
    random.shuffle(shuffle)

    merged_imgs = [merged_img_1, merged_img_2]
    shuffled_imgs = [merged_imgs[shuffle[0]], merged_imgs[shuffle[1]]]
    merged_img = np.maximum.reduce([shuffled_imgs[0] * 0.4, shuffled_imgs[1]]).astype(np.uint8)

    merged_labels = [merged_label_1, merged_label_2]
    shuffled_labels = [merged_labels[shuffle[0]], merged_labels[shuffle[1]]]
    merged_label = np.maximum.reduce([shuffled_labels[0] * 0.4, shuffled_labels[1]])
    if shuffle[0] == 0:
        merged_label = np.where(merged_label == 0.4, 1, merged_label).astype(np.uint8)
    else:
        merged_label = np.where(merged_label == 0.8, 2, merged_label).astype(np.uint8)

    return merged_img, merged_label, merged_img_1, merged_label_1, merged_img_2, merged_label_2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, images_1, labels_1, images_2, labels_2, index):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.dest, str(index) + '.tfrecords')
    print('Writing: %s, images num: %d' % (filename, len(images)))
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(len(images)):
        image_raw = images[i].tostring()
        label_raw = labels[i].tostring()
        image_raw_1 = images_1[i].tostring()
        label_raw_1 = labels_1[i].tostring()
        image_raw_2 = images_2[i].tostring()
        label_raw_2 = labels_2[i].tostring()

        features = tf.train.Features(feature={
            'height': _int64_feature(images[i].shape[0]),
            'width': _int64_feature(images[i].shape[1]),
            'depth': _int64_feature(1),
            'label_1': _int64_feature(1),
            'label_2': _int64_feature(2),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw),
            'image_raw_1': _bytes_feature(image_raw_1),
            'label_raw_1': _bytes_feature(label_raw_1),
            'image_raw_2': _bytes_feature(image_raw_2),
            'label_raw_2': _bytes_feature(label_raw_2)})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def main(unused_argv):
    input_folder1 = FLAGS.data_dir1
    input_folder2 = FLAGS.data_dir2
    images_num = FLAGS.image_num
    height, width = FLAGS.height, FLAGS.width

    start = 1
    end = min(FLAGS.file_size, images_num + 1)
    fidx = 0
    while True:
        images, labels = [], []
        images_1, labels_1 = [], []
        images_2, labels_2 = [], []
        for i in range(start, end):
            file_path_1 = os.path.join(input_folder1, 'annotation_' + str(i).zfill(4) + '.mat')
            file_path_2 = os.path.join(input_folder2, 'annotation_' + str(i).zfill(4) + '.mat')
            img, label, img_1, label_1, img_2, label_2 = merge_images(file_path_1, file_path_2, height, width)
            # cv2.imwrite(os.path.join(FLAGS.dest, str(i) + '.png'), img)
            # cv2.imwrite(os.path.join(FLAGS.dest, str(i) + '_l.png'), label * 100)

            images.append(img), labels.append(label)
            images_1.append(img_1), labels_1.append(label_1)
            images_2.append(img_2), labels_2.append(label_2)

        convert(images, labels, images_1, labels_1, images_2, labels_2, fidx)
        if end < (images_num + 1):
            start = end
            end = min(end + FLAGS.file_size, images_num + 1)
            fidx += 1
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir1',
        type=str,
        default='/tmp/data',
        help='Directory to download data files.'
    )
    parser.add_argument(
        '--data_dir2',
        type=str,
        default='/tmp/data',
        help='Directory to download data files.'
    )
    parser.add_argument(
        '--image_num',
        type=int,
        default='89',
        help='Total file number.'
    )
    parser.add_argument(
        '--height',
        type=int,
        default='28',
        help='Height of generated images.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default='28',
        help='Width of generated images.'
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
        default='20',
        help='The volume of each generated file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
