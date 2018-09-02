from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from python.layers.convolution import conv2d, deconv
from python.layers.primary_capsules import primary_caps1d
from python.layers.conv_capsules import conv_capsule1d
from python.layers.class_capsules import class_caps1d


def camvid_inference(inputs, num_classes, routing_ites=3, remake=False, name='capsnet_1d'):
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

        print("\nconv0 layer:")
        conv0 = conv2d(
            inputs,
            kernel=9, out_channels=32, stride=2, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv0'
        )
        # conv1 = tf.check_numerics(conv1, message="nan or inf from: conv1")
        print('conv0 shape: %s' % conv0.get_shape())

        print("\nconv1 layer:")
        conv1 = conv2d(
            conv0,
            kernel=5, out_channels=64, stride=2, padding='VALID',
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
        print('conv2 shape: %s' % conv2.get_shape())
        # primary_caps_activations = tf.check_numerics(primary_caps_activations, message="nan or inf from: primary_caps_activations")
        # primary_caps_activations = tf.Print(primary_caps_activations, [tf.constant("primary_caps_activations"), primary_caps_activations])

        # print("\nconv capsule layer:")
        # conv_out_capsules = 32
        # conv_caps_activations = conv_capsule1d(
        #     primary_caps_activations, kernel_size=3,
        #     stride=1, routing_ites=routing_ites, in_capsules=primary_out_capsules,
        #     out_capsules=conv_out_capsules, batch_size=batch_size, name='conv_caps'
        # )  # (b, 32*4*20, 8)
        # conv_caps_activations = tf.Print(conv_caps_activations, [tf.constant("conv_caps_activations"), conv_caps_activations])

        # (b, 32, 4, 20, 8) -> # (b, 32*4*20, 2*64)
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
            num_classes=num_classes, batch_size=batch_size, conv0=conv0, conv1=conv1, conv2=conv2)
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

def _decode(activations, capsule_num, coupling_coeffs, num_classes, batch_size, conv0, conv1, conv2):
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
    concat1 = tf.concat([tf.transpose(conv2, perm=[0, 2, 3, 1]), primary_labels], axis=3, name='concat1')
    primary_conv = conv2d(
        concat1,
        kernel=3, out_channels=128, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='primary_conv'
    )

    deconv2 = deconv(
        primary_conv,
        kernel=6, out_channels=64, stride=2,
        activation_fn=tf.nn.relu, name='deconv2'
    )
    print('deconv2 shape: %s' % deconv2.get_shape())
    concat2 = tf.concat([tf.transpose(conv1, perm=[0, 2, 3, 1]), deconv2], axis=3, name='concat2')
    # deconv2 = tf.Print(deconv2, [tf.constant("deconv2"), deconv2])
    deconv2_conv = conv2d(
        concat2,
        kernel=3, out_channels=32, stride=1, padding='SAME',
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
        kernel=6, out_channels=32, stride=2,
        activation_fn=tf.nn.relu, name='deconv3'
    )
    print('deconv3 shape: %s' % deconv3.get_shape())
    concat3 = tf.concat([tf.transpose(conv0, perm=[0, 2, 3, 1]), deconv3], axis=3, name='concat3')
    deconv3_conv = conv2d(
        concat3,
        kernel=3, out_channels=32, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='deconv3_conv'
    )

    deconv4 = deconv(
        deconv3_conv,
        kernel=10, out_channels=num_classes, stride=2,
        activation_fn=tf.nn.relu, name='deconv4'
    )
    deconv4_conv = conv2d(
        deconv4,
        kernel=3, out_channels=num_classes, stride=1, padding='SAME',
        activation_fn=tf.nn.relu, data_format='NHWC', name='deconv4_conv'
    )

    label_logits = deconv4_conv
    print('label_logits shape: %s' % label_logits.get_shape())
    # label_logits = tf.Print(label_logits, [tf.constant("label_logits"), label_logits])
    return label_logits