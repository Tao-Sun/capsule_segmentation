from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from python.layers.convolution import conv2d, deconv
from python.layers.primary_capsules import primary_caps1d
from python.layers.conv_capsules import conv_capsule1d
from python.layers.class_capsules import class_caps1d
import python.data.affnist.affnist_input as affnist_input

import numpy as np


data_input = affnist_input

def inference(inputs, num_classes, routing_ites=3, remake=False, training=False, name='capsnet_1d'):
    """

    :param inputs:
    :param num_classes:
    :param routing_ites:
    :param remake:
    :param name:
    :return:
    """

    with tf.variable_scope(name) as scope:
        inputs_shape = inputs.get_shape()
        batch_size = inputs_shape[0].value
        image_height = inputs_shape[2].value
        image_width = inputs_shape[3].value

        # ReLU Conv1
        # Images shape (b, 1, 24, 56) -> conv 5x5 filters, 32 output channels, strides 2 with padding, ReLU
        # nets -> (b, 256, 16, 48)
        print('inputs shape: %s' % inputs.get_shape())
        inputs = tf.check_numerics(inputs, message="nan or inf from: inputs")

        print("\nconv1 layer:")
        conv1 = conv2d(
            inputs,
            kernel=9, out_channels=256, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv1'
        )
        # conv1 = tf.check_numerics(conv1, message="nan or inf from: conv1")
        print('conv1 shape: %s' % conv1.get_shape())

        # print("\nconv2 layer:")
        # conv2 = conv2d(
        #     conv1,
        #     kernel=5, out_channels=256, stride=1, padding='VALID',
        #     activation_fn=tf.nn.relu, name='relu_conv2'
        # )
        # # conv2 = tf.check_numerics(conv2, message="nan or inf from: conv2")
        # print('conv2 shape: %s' % conv2.get_shape())

        # PrimaryCaps
        # (b, 256, 16, 48) -> capsule 1x1 filter, 32 output capsule, strides 1 without padding
        # nets -> activations (?, 14, 14, 32))
        print("\nprimary layer:")
        primary_out_capsules = 24
        primary_caps_activations, conv2 = primary_caps1d(
            conv1,
            kernel_size=5, out_capsules=primary_out_capsules, stride=2,
            padding='VALID', activation_length=8, name='primary_caps'
        )  # (b, 32, 4, 20, 8)


        print("\nconvolutional capsule layer:")
        conv_out_capsules = 24
        conv_kernel_size, conv_stride = 3, 1
        conv_caps_activations, conv_coupling_coeffs = conv_capsule1d(
            primary_caps_activations,
            kernel_size=conv_kernel_size, stride=conv_stride, routing_ites=3,
            in_capsules=primary_out_capsules, out_capsules=conv_out_capsules,
            activation_length=8, name="conv_caps"
        )  # (b, 32, 6, 6, 8), (b*6*6, 32*9, 32)

        # (b, 32, 4, 20, 8) -> # (b, 32*4*20, 2*64)
        print("\nclass capsule layer:")
        class_caps_activations, class_coupling_coeffs = class_caps1d(
            conv_caps_activations,
            num_classes=num_classes, activation_length=16, routing_ites=routing_ites,
            batch_size=batch_size, name='class_capsules')
        # class_coupling_coeffs = tf.Print(class_coupling_coeffs, [class_coupling_coeffs], summarize=50)
        # class_caps_activations = tf.check_numerics(class_caps_activations, message="nan or inf from: class_caps_activations")
        print('class_coupling_coeffs shape: %s' % class_coupling_coeffs.get_shape())    # (b, 32*4*20, 2)
        print('class_caps_activations shape: %s' % class_caps_activations.get_shape())  # (b, 2, 64)

        if remake:
            remakes_flatten = _remake(class_caps_activations, image_height * image_width)
        else:
            remakes_flatten = None

        print("\ndecode layers:")
        label_logits = _decode(
            conv_caps_activations, primary_caps_activations,
            class_coupling_coeffs, conv_coupling_coeffs,
            conv_kernel_size, conv_stride, primary_out_capsules,
            num_classes=num_classes, batch_size=batch_size, conv1=conv1, conv2=conv2)
        # label_logits = tf.Print(label_logits, [tf.constant("label_logits"), label_logits[0]], summarize=100)
        # label_logits = tf.check_numerics(label_logits, message="nan or inf from: label_logits")

        labels2d = tf.argmax(label_logits, axis=3)
        labels2d_expanded = tf.expand_dims(labels2d, -1)
        tf.summary.image('labels', tf.cast(labels2d_expanded, tf.uint8))

    return class_caps_activations, remakes_flatten, label_logits


def _remake(class_caps_activations, num_pixels):
    first_layer_size, second_layer_size = 512, 1024
    capsules_2d = tf.contrib.layers.flatten(class_caps_activations)

    remakes_flatten = tf.contrib.layers.stack(
        inputs=capsules_2d,
        layer=tf.contrib.layers.fully_connected,
        stack_args=[(first_layer_size, tf.nn.relu),
                    (second_layer_size, tf.nn.relu),
                    (num_pixels, tf.sigmoid)],
        reuse=False, scope='remake',
        weights_initializer=tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32),
        biases_initializer=tf.constant_initializer(0.1))

    return remakes_flatten  # (b, 1344)


def _decode(conv_activations, primary_caps_activations, class_coupling_coeffs, conv_coupling_coeffs,
            conv_kernel_size, conv_stride, primary_out_capsules, num_classes, batch_size, conv1, conv2):
    # capsule_probs = tf.norm(activations, axis=-1)  # # (b, 32, 4, 20, 8) -> (b, 32, 4, 20)
    # caps_probs_tiled = tf.tile(tf.expand_dims(class_coupling_coeffs, 1), [1, 1, 1, 1, num_classes])  # (b, 32, 4, 20, 2)
    # caps_probs_tiled = tf.check_numerics(caps_probs_tiled, message="nan or inf from: caps_probs_tiled")

    print("\ntraceback layer:")
    print('class coupling_coeffs shape: %s' % class_coupling_coeffs.get_shape())  # (b, 32*6*6, 2)
    activations_shape = conv_activations.get_shape()  # (b, 32, 4, 20, 8),
    conv_capsule_num, height, width = activations_shape[1].value, activations_shape[2].value, activations_shape[3].value
    cls_coupling_coeff_reshaped = tf.reshape(class_coupling_coeffs, [batch_size, conv_capsule_num, height, width, num_classes])  # (b, 32, 6, 6, 2)
    cls_coupling_coeff_expanded = tf.expand_dims(tf.expand_dims(cls_coupling_coeff_reshaped, 4), 5)  # (b, 32, 6, 6, 1, 1, 2)
    cls_coupling_coeff_transposed = tf.transpose(cls_coupling_coeff_expanded, perm=[0, 2, 3, 4, 5, 1, 6])  # (b, 6, 6, 1, 1, 32, 2)
    cls_coupling_coeff_tiled = tf.tile(cls_coupling_coeff_transposed,
                                   [1, 1, 1, conv_kernel_size, conv_kernel_size, 1, 1])  # (b, 6, 6, 3, 3, 32, 2)
    print('cls_coupling_coeff_tiled shape: %s' % cls_coupling_coeff_tiled.get_shape())

    print('conv coupling_coeffs shape: %s' % conv_coupling_coeffs.get_shape())  # (b*6*6, 32*3*3, 32)
    conv_coupling_coeff_reshaped = tf.reshape(conv_coupling_coeffs,
                                             [batch_size, height, width, primary_out_capsules,
                                              conv_kernel_size, conv_kernel_size, conv_capsule_num])  #(b, 6, 6, 32, 3, 3, 32)
    conv_coupling_coeff_transposed = tf.transpose(conv_coupling_coeff_reshaped, perm=[0, 1, 2, 4, 5, 3, 6])  #(b, 6, 6, 3, 3, 32, 32)
    print('conv_coupling_coeff_transposed shape: %s' % conv_coupling_coeff_transposed.get_shape())  #(b, 6, 6, 3, 3, 32, 32)

    primary_cond_prob_stacked = tf.matmul(conv_coupling_coeff_transposed, cls_coupling_coeff_tiled)  #(b, 6, 6, 3, 3, 32, 2)
    primary_cond_prob_stacked = tf.reshape(primary_cond_prob_stacked,
                                            [batch_size, height, width, conv_kernel_size*conv_kernel_size,
                                             primary_out_capsules*num_classes])  # (b, 6, 6, 3*3, 32*2)
    primary_cond_prob_stacked = tf.transpose(primary_cond_prob_stacked, perm=[0, 1, 2, 4, 3])  # (b, 6, 6, 32*2, 3*3)
    print('primary_cond_prob_stacked shape: %s' % primary_cond_prob_stacked.get_shape())

    tile_filter = np.zeros(shape=[conv_kernel_size, conv_kernel_size, 1,
                                  1, conv_kernel_size*conv_kernel_size], dtype=np.float32)
    for i in range(conv_kernel_size):
        for j in range(conv_kernel_size):
            tile_filter[i, j, 0, 0, i*conv_kernel_size+j] = 1.0  # (3, 3, 1, 1, 9)
    # (3, 3, 1, 9, 1)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    out_h = (height - 1) * conv_stride + conv_kernel_size
    out_w = (width - 1) * conv_stride + conv_kernel_size
    output_shape = tf.constant([batch_size, out_h, out_w, primary_out_capsules*num_classes, 1], dtype=tf.int32)
    primary_cond_prob = tf.nn.conv3d_transpose(primary_cond_prob_stacked, tile_filter_op, output_shape,
                                               strides=[1, conv_stride, conv_stride, 1, 1], padding='VALID')  # (b, 14, 14, 32*2, 1)
    print('primary_cond_prob shape: %s' % primary_cond_prob.get_shape())
    primary_cond_prob_shape = primary_cond_prob.get_shape()  # (b, 14, 14, 32*2, 1)
    height, width = primary_cond_prob_shape[1].value, primary_cond_prob_shape[2].value
    primary_cond_prob_reshaped = tf.reshape(primary_cond_prob, [batch_size, height, width, primary_out_capsules, num_classes])  # (b, 14, 14, 32, 2)
    primary_cond_prob_transposed = tf.transpose(primary_cond_prob_reshaped, perm=[0, 3, 1, 2, 4])  # (b, 32, 14, 14, 2)

    capsule_probs = tf.norm(primary_caps_activations, axis=-1)  # # (b, 32, 14, 14, 8) -> (b, 32, 14, 14)
    caps_probs_tiled = tf.tile(tf.expand_dims(capsule_probs, -1), [1, 1, 1, 1, num_classes])  # (b, 32, 14, 14, 2)

    primary_labels = tf.reduce_sum(primary_cond_prob_transposed*caps_probs_tiled, 1)  # (b, 4, 20, 2)

    print('primary_caps_labels shape: %s' % primary_labels.get_shape())
    # class_labels = tf.Print(class_labels, [tf.constant("class_labels"), class_labels])
    # class_labels = tf.check_numerics(class_labels, message="nan or inf from: class_labels")
    # primary_labels = tf.reduce_sum(caps_probs_tiled, 1)

    concat1 = tf.concat([tf.transpose(conv2, perm=[0, 2, 3, 1]), primary_labels], axis=3, name='concat1')
    primary_conv = conv2d(
        concat1,
        kernel=3, out_channels=128, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='primary_conv'
    )

    deconv2 = deconv(
        primary_conv,
        kernel=6, out_channels=128, stride=2,
        activation_fn=tf.nn.relu, name='deconv2'
    )
    print('deconv2 shape: %s' % deconv2.get_shape())
    concat2 = tf.concat([tf.transpose(conv1, perm=[0, 2, 3, 1]), deconv2], axis=3, name='concat2')
    # deconv2 = tf.Print(deconv2, [tf.constant("deconv2"), deconv2])
    deconv2_conv = conv2d(
        concat2,
        kernel=3, out_channels=128, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='deconv2_conv'
    )
    print('deconv2_conv shape: %s' % deconv2_conv.get_shape())

    # deconv3 = deconv(
    #     class_labels,
    #     kernel=9, out_channels=num_classes, stride=1,
    #     activation_fn=tf.nn.relu, name='deconv3'
    # )
    # deconv3 = tf.Print(deconv3, [tf.constant("deconv3"), deconv3])
    deconv3 = deconv(
        deconv2_conv,
        kernel=9, out_channels=num_classes, stride=1,
        activation_fn=tf.nn.relu, name='deconv3'
    )
    deconv3_conv = conv2d(
        deconv3,
        kernel=3, out_channels=num_classes, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='deconv3_conv'
    )

    label_logits = deconv3_conv
    print('label_logits shape: %s' % label_logits.get_shape())
    # label_logits = tf.Print(label_logits, [tf.constant("label_logits"), label_logits])
    return label_logits


def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.
    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.
    Args:
      labels: tensor, one hot encoding of ground truth.
      raw_logits: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.
    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    logits = raw_logits - 0.5
    positive_cost = labels * \
        tf.cast(tf.less(logits, margin), tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def loss(images, labels2d, class_caps_activations, remakes_flatten, label_logits, label_class, num_classes):
    """

    :param images:
    :param labels2d:
    :param class_caps_activations:
    :param remakes_flatten:
    :param label_logits:
    :param num_classes:
    :return:
    """

    with tf.name_scope('loss'):
        if remakes_flatten is not None:
            with tf.name_scope('remake'):
                image_flatten = tf.contrib.layers.flatten(images)
                distance = tf.pow(image_flatten - remakes_flatten, 2)
                remake_loss = tf.reduce_sum(distance, axis=-1)

                batch_remake_loss = tf.reduce_mean(remake_loss)
                balanced_remake_loss = 0.05 * batch_remake_loss

                tf.add_to_collection('losses', balanced_remake_loss)
                tf.summary.scalar('remake_loss', balanced_remake_loss)

        with tf.name_scope('margin'):
            one_hot_label_class = label_class  # tf.one_hot(label_class, depth=num_classes)
            class_caps_logits = tf.norm(class_caps_activations, axis=-1)
            margin_loss = _margin_loss(one_hot_label_class, class_caps_logits)

            batch_margin_loss = tf.reduce_mean(margin_loss)
            balanced_margin_loss = batch_margin_loss
            # batch_margin_loss = tf.Print(batch_margin_loss, [batch_margin_loss])
            tf.add_to_collection('losses', balanced_margin_loss)
            tf.summary.scalar('margin_loss', balanced_margin_loss)

        with tf.name_scope('decode'):
            # labels2d = tf.Print(labels2d, [labels2d[0]], summarize=100, message="labels2d: ")
            # label_logits = tf.Print(label_logits, [label_logits[0]], message="label_logits: ")
            one_hot_labels = tf.one_hot(labels2d, depth=num_classes)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,
                                                                    logits=label_logits)

            # class_weights = tf.constant([1.0, 8.0, 2.0, 2.0, 10.0])
            class_weights = tf.constant([1.0] + [5.0] * (num_classes - 1))
            # deduce weights for batch samples based on their true label
            weights = tf.reduce_sum(class_weights * one_hot_labels, axis=3)

            weighted_losses = cross_entropy * weights
            print('labels2d shape: %s' % labels2d.get_shape())
            print('label_logits shape: %s' % label_logits.get_shape())
            print('cross_entropy shape: %s' % weighted_losses.get_shape())

            batch_decode_loss = tf.reduce_mean(weighted_losses)
            balanced_decode_loss = batch_decode_loss

            tf.add_to_collection('losses', balanced_decode_loss)
            tf.summary.scalar('decode_loss', balanced_decode_loss)

def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=1000,
        learning_rate=0.0005,
        momentum=0.99
    )
