from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


"""
trace_conv_cond_prob: calculate P(C_k|i)
trace_labels: calculate P(C_k)
"""


def trace_labels(caps_activations, cond_prob, num_classes):
    # (b, 32, 14, 14, 8) -> (b, 32, 14, 14)
    capsule_probs = tf.norm(caps_activations, axis=-1)
    # (b, 32, 14, 14, 2)
    caps_probs_tiled = tf.tile(tf.expand_dims(capsule_probs, -1), [1, 1, 1, 1, num_classes])
    print('caps_probs_tiled shape: %s' % caps_probs_tiled.get_shape())

    capsule_probs_shape = caps_probs_tiled.get_shape()  # (b, 32, 14, 14, 2)
    batch_size = capsule_probs_shape[0].value
    capsule_num = capsule_probs_shape[1].value
    height, width = capsule_probs_shape[2].value, capsule_probs_shape[3].value
    # (b, 32, 14, 14, 2)
    cond_prob_reshaped = tf.reshape(cond_prob,  # (b, 32*14*14, 2)
                                    [batch_size, capsule_num,
                                     height, width, num_classes])
    print('cond_prob_reshaped shape: %s' % cond_prob_reshaped.get_shape())

    # (b, 4, 20, 2)
    labels = tf.reduce_sum(cond_prob_reshaped*caps_probs_tiled, 1)
    print('lables shape: %s' % labels.get_shape())

    return labels


def trace_conv_cond_prob(prev_cond_prob, conv_coupling_coeffs,
                         height, width, kernel_size, stride,
                         prev_capsule_num, capsule_num):
    """
    With the conditional probabilities P(C_k|j) for previous layer,
    calculate the conditional probabilities P(C_k|i) for current
    convolutional capsule layer. Note that the conditional probabilities
    P(C_k|j) for L-1 layer is just its coupling coefficients.
    Steps:
        1. Calculate the conditional probabilities P(C_k|i) and stack
           them in the previous layer.
        2. Unstack the conditional probabilities P(C_k|i) and distribute
           them in the current layer through a deconvolutional operation.
    """

    batch_size = prev_cond_prob.get_shape()[0].value
    num_classes = prev_cond_prob.get_shape()[2].value

    # (b, 6, 6, 3*3, 32*2)
    stacked_cond_prob = _get_stacked_conv_cond_prob(prev_cond_prob, conv_coupling_coeffs,
                                                    height, width, kernel_size,
                                                    prev_capsule_num, capsule_num)
    # (b, 14, 14, 32*2, 1)
    unstacked_conv_cond_prob = _unstack_conv_cond_prob(stacked_cond_prob, capsule_num,
                                                     height, width, kernel_size, stride,
                                                     batch_size, num_classes)

    cond_prob_shape = unstacked_conv_cond_prob.get_shape()  # (b, 14, 14, 32*2, 1)
    height, width = cond_prob_shape[1].value, cond_prob_shape[2].value
    # (b, 14, 14, 32, 2)
    cond_prob_reshaped = tf.reshape(unstacked_conv_cond_prob,
                                    [batch_size, height, width,
                                     capsule_num, num_classes])
    # (b, 32*14*14, 2)
    cond_prob = tf.reshape(tf.transpose(cond_prob_reshaped, perm=[0, 3, 1, 2, 4]),   # (b, 32, 14, 14, 2),
                           [batch_size, capsule_num*height*width, num_classes])
    print('cond_prob shape: %s' % cond_prob.get_shape())  # (b, 32*14*14, 2)

    return cond_prob


def _get_stacked_conv_cond_prob(prev_cond_prob, conv_coupling_coeffs,
                                height, width, kernel_size,
                                prev_capsule_num, capsule_num):
    """
    As stated in the paper, tracing back for each capsule of current
    layers is actually a matrix multiplication operation. And so conditional
    probabilities are calculated through matrix multiplication operations.
    and stacked in the previous layer.
    """
    batch_size = prev_cond_prob.get_shape()[0].value
    num_classes = prev_cond_prob.get_shape()[2].value

    print('previous coupling_coeffs shape: %s' % prev_cond_prob.get_shape())  # (b, 32*6*6, 2)
    # (b, 32, 6, 6, 2)
    prev_cond_prob_reshaped = tf.reshape(prev_cond_prob,
                                         [batch_size, prev_capsule_num,
                                          height, width, num_classes])

    # (b, 6, 6, 1, 1, 32, 2)
    prev_cond_prob_transposed = tf.transpose(tf.expand_dims(tf.expand_dims(prev_cond_prob_reshaped, 4), 5),
                                             # (b, 32, 6, 6, 1, 1, 2)
                                             perm=[0, 2, 3, 4, 5, 1, 6])
    # (b, 6, 6, 3, 3, 32, 2)
    prev_cond_prob_tiled = tf.tile(prev_cond_prob_transposed,
                                   [1, 1, 1, kernel_size,
                                    kernel_size, 1, 1])
    print('prev_cond_prob_tiled shape: %s' % prev_cond_prob_tiled.get_shape())

    print('conv coupling_coeffs shape: %s' % conv_coupling_coeffs.get_shape())  # (b*6*6, 32*3*3, 32)
    # (b, 6, 6, 32, 3, 3, 32)
    conv_coupling_coeffs_reshaped = tf.reshape(conv_coupling_coeffs,
                                               [batch_size, height, width, capsule_num,
                                                kernel_size, kernel_size, prev_capsule_num])
    # (b, 6, 6, 3, 3, 32, 32)
    conv_coupling_coeffs_transposed = tf.transpose(conv_coupling_coeffs_reshaped,
                                                   perm=[0, 1, 2, 4, 5, 3, 6])
    print(
        'coupling_coeffs_transposed shape: %s' % conv_coupling_coeffs_transposed.get_shape())  # (b, 6, 6, 3, 3, 32, 32)

    # ((b, 6, 6, 3, 3, 32, 2) = (b, 6, 6, 3, 3, 32, 32) * (b, 6, 6, 3, 3, 32, 2))
    stacked_cond_prob = tf.matmul(conv_coupling_coeffs_transposed, prev_cond_prob_tiled)
    # (b, 6, 6, 3*3, 32*2)
    stacked_cond_prob = tf.transpose(tf.reshape(stacked_cond_prob,
                                                [batch_size, height, width,
                                                 kernel_size * kernel_size,
                                                 capsule_num * num_classes]),  # (b, 6, 6, 32*2, 3*3)
                                     perm=[0, 1, 2, 4, 3])
    print('stacked_cond_prob shape: %s' % stacked_cond_prob.get_shape())  # (b, 6, 6, 3*3, 32*2)

    return stacked_cond_prob


def _unstack_conv_cond_prob(stacked_cond_prob, capsule_num,
                            height, width, kernel_size, stride,
                            batch_size, num_classes):
    """
    Unstack the conditional probabilities P(C_k|i) and distribute
    them in the current layer through a deconvolutional operation.
    """

    tile_filter = np.zeros(shape=[kernel_size, kernel_size, 1, 1, kernel_size*kernel_size], dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            tile_filter[i, j, 0, 0, i*kernel_size+j] = 1.0  # (3, 3, 1, 1, 9)
    # (3, 3, 1, 9, 1)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    out_h = (height - 1) * stride + kernel_size
    out_w = (width - 1) * stride + kernel_size
    output_shape = tf.constant([batch_size, out_h, out_w, capsule_num*num_classes, 1], dtype=tf.int32)
    # (b, 14, 14, 32*2, 1)
    unstacked_cond_prob = tf.nn.conv3d_transpose(stacked_cond_prob,  # (b, 6, 6, 3*3, 32*2)
                                                 tile_filter_op, output_shape,
                                                 strides=[1, stride, stride, 1, 1],
                                                 padding='VALID')
    print('unstacked_cond_prob shape: %s' % unstacked_cond_prob.get_shape())  # (b, 14, 14, 32*2, 1)

    return unstacked_cond_prob
