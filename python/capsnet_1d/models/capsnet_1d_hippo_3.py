from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from python.layers.convolution import conv2d, deconv
from python.layers.primary_capsules import primary_caps1d
from python.layers.conv_capsules import conv_capsule1d
from python.layers.class_capsules import class_caps1d
import python.data.hippo.hippo_input as hippo_input


data_input = hippo_input


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

        conv1 = conv2d(
            inputs,
            kernel=3, out_channels=32, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='relu_conv1'
        )
        print('conv1 shape: %s' % conv1.get_shape())
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
            padding='VALID', data_format='NCHW', name='pool1'
        )
        print('pool1 shape: %s' % pool1.get_shape())

        conv2 = conv2d(
            pool1,
            kernel=5, out_channels=64, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='relu_conv2'
        )
        print('conv2 shape: %s' % conv2.get_shape())
        # pool2 = tf.nn.max_pool(
        #     conv2,
        #     ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        #     padding='VALID', data_format='NCHW', name='pool2'
        # )
        # print('pool2 shape: %s' % pool2.get_shape())

        # conv3 = conv2d(
        #     pool2,
        #     kernel=3, out_channels=128, stride=1, padding='SAME',
        #     activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        #     name='relu_conv3'
        # )
        # print('conv3 shape: %s' % conv3.get_shape())
        # pool3 = tf.nn.max_pool(
        #     conv3,
        #     ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
        #     padding='VALID', data_format='NCHW', name='pool3'
        # )
        # print('pool3 shape: %s' % pool3.get_shape())


        print("\nprimary layer:")
        primary_out_capsules = 32
        primary_caps_activations, conv_primary = primary_caps1d(
            conv2,
            kernel_size=5, out_capsules=primary_out_capsules, stride=1,
            padding='VALID', activation_length=8, name='primary_caps'
        )  # (b, 32, 4, 20, 8)

        print("\nclass capsule layer:")
        class_caps_activations, class_coupling_coeffs = class_caps1d(
            primary_caps_activations,
            num_classes=num_classes, activation_length=16, routing_ites=routing_ites,
            batch_size=batch_size, name='class_capsules')
        # class_coupling_coeffs = tf.Print(class_coupling_coeffs, [class_coupling_coeffs], summarize=50)
        # class_caps_activations = tf.check_numerics(class_caps_activations, message="nan or inf from: class_caps_activations")
        print('class_coupling_coeffs shape: %s' % class_coupling_coeffs.get_shape())
        print('class_caps_activations shape: %s' % class_caps_activations.get_shape())

        if remake:
            remakes_flatten = _remake(class_caps_activations, image_height * image_width)
        else:
            remakes_flatten = None

        print("\ndecode layers:")
        label_logits = _decode(
            primary_caps_activations, primary_out_capsules,
            coupling_coeffs=class_coupling_coeffs,
            num_classes=num_classes, batch_size=batch_size,
            pool1=pool1, conv2=conv2, training=training)
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

def _decode(activations, capsule_num, coupling_coeffs, num_classes, batch_size, pool1, conv2, training):
    capsule_probs = tf.norm(activations, axis=-1)  # # (b, 32, 4, 20, 8) -> (b, 32, 4, 20)
    caps_probs_tiled = tf.tile(tf.expand_dims(capsule_probs, -1), [1, 1, 1, 1, num_classes])  # (b, 32, 4, 20, 2)
    # caps_probs_tiled = tf.check_numerics(caps_probs_tiled, message="nan or inf from: caps_probs_tiled")

    print('coupling_coeffs shape: %s' % coupling_coeffs.get_shape())
    activations_shape = activations.get_shape()
    height, width = activations_shape[2].value, activations_shape[3].value
    coupling_coeff_reshaped = tf.reshape(coupling_coeffs, [batch_size, capsule_num, height, width, num_classes])  # (b, 32, 4, 20, 2)
    # coupling_coeff_reshaped = tf.check_numerics(coupling_coeff_reshaped, message="nan or inf from: coupling_coeff_reshaped")

    primary_labels = tf.reduce_sum(coupling_coeff_reshaped * caps_probs_tiled, 1)  # (b, 4, 20, 2)
    # class_labels = tf.Print(class_labels, [tf.constant("class_labels"), class_labels])
    # class_labels = tf.check_numerics(class_labels, message="nan or inf from: class_labels")
    # primary_labels = tf.reduce_sum(caps_probs_tiled, 1)
    # deconv1 = deconv(
    #     class_labels,
    #     kernel=3, out_channels=num_classes, stride=1,
    #     activation_fn=tf.nn.relu, name='deconv1'
    # )
    # deconv1 = tf.Print(deconv1, [tf.constant("deconv1"), deconv1])
    print('primary_labels shape: %s' % primary_labels.get_shape())
    primary_conv = conv2d(
        tf.transpose(primary_labels, perm=[0, 3, 1, 2]),
        kernel=3, out_channels=256, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, name='primary_conv'
    )
    print('primary_conv shape: %s' % primary_conv.get_shape())

    deconv1 = deconv(
        primary_conv,
        kernel=5, out_channels=128, stride=1, data_format='NCHW',
        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        name='deconv1'
    )
    print('deconv1 shape: %s' % deconv1.get_shape())
    # deconv1_conv = conv2d(
    #     deconv1,
    #     kernel=3, out_channels=128, stride=1, padding='SAME',
    #     activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
    #     name='deconv1_conv'
    # )
    # print('deconv1_conv shape: %s' % deconv1_conv.get_shape())
    concat1 = tf.concat([conv2, deconv1], axis=1, name='concat1')
    dropout1 = tf.layers.dropout(concat1, 0.5, training=training, name='dropout1')
    concat1_conv = conv2d(
        dropout1,
        kernel=3, out_channels=128, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        name='concat1_conv'
    )
    print('concat1_conv shape: %s' % concat1_conv.get_shape())

    deconv2 = deconv(
        concat1_conv,
        kernel=5, out_channels=128, stride=1, data_format='NCHW',
        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        name='deconv2'
    )
    print('deconv2 shape: %s' % deconv2.get_shape())
    # deconv2_conv = conv2d(
    #     deconv2,
    #     kernel=3, out_channels=128, stride=1, padding='SAME',
    #     activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
    #     name='deconv2_conv'
    # )
    # print('deconv2_conv shape: %s' % deconv2_conv.get_shape())
    concat2 = tf.concat([pool1, deconv2], axis=1, name='concat2')
    dropout2 = tf.layers.dropout(concat2, 0.5, training=training, name='dropout2')
    concat2_conv = conv2d(
        dropout2,
        kernel=3, out_channels=128, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        name='concat2_conv'
    )
    print('concat2_conv shape: %s' % concat2_conv.get_shape())

    deconv3 = deconv(
        concat2_conv,
        kernel=4, out_channels=128, stride=2, data_format='NCHW',
        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
        name='deconv3'
    )
    print('deconv3 shape: %s' % deconv3.get_shape())

    class_conv = conv2d(
        deconv3,
        kernel=3, out_channels=num_classes, stride=1, padding='VALID',
        name='class_conv'
    )
    print('class_conv shape: %s' % class_conv.get_shape())

    label_logits = tf.transpose(class_conv, perm=[0, 2, 3, 1])
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
            balanced_margin_loss = 1.5 * batch_margin_loss
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
            balanced_decode_loss = 0.1 * batch_decode_loss

            tf.add_to_collection('losses', balanced_decode_loss)
            tf.summary.scalar('decode_loss', balanced_decode_loss)

def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.96,
        decay_steps=1000,
        learning_rate=0.001
    )
