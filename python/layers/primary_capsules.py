import tensorflow as tf

from python.layers.routing import dynamic_routing
from .convolution import conv2d


def primary_caps1d(inputs, kernel_size, out_capsules, stride, padding, activation_length, name):
    """

    :param inputs:
    :param kernel_size:
    :param out_capsules:
    :param stride:
    :param padding:
    :param activation_length:
    :param name:
    :return:
    """

    with tf.variable_scope(name):
        conv = conv2d(
            inputs,
            kernel=kernel_size, out_channels=out_capsules * activation_length,
            stride=stride, padding=padding, name='primary_caps_conv',
            normalizer_fn=tf.contrib.layers.batch_norm,
        )  # (b, 256, 16, 48) -> (b, 256, 4, 20)]
        print('primary conv shape: %s' % conv.get_shape())

        conv_shape = conv.get_shape()
        conv_height, conv_width = conv_shape[2].value, conv_shape[3].value
        conv_reshaped = tf.reshape(conv,
                                   [-1, 1, out_capsules, activation_length, conv_height, conv_width])
        conv_reshaped = tf.check_numerics(conv_reshaped, message="nan or inf from: convolution in primary layer")
        # conv_reshaped = tf.Print(conv_reshaped, [tf.constant("conv_reshaped"), conv_reshaped[0][0][0][0][0:24][0:24]], summarize=3)
        print('votes shape: %s' % conv_reshaped.get_shape())

        with tf.name_scope('routing'):
            activations, _ = dynamic_routing(
                votes=conv_reshaped,
                coupling_coeffs_shape=tf.stack([conv_shape[0], 1, out_capsules, conv_height, conv_width]),
                num_dims=6,
                input_dim=1,
                num_routing=1,
                caller=" primary")
            # activations = tf.check_numerics(activations, message="nan or inf from: activations in primary layer")
            activations_transposed = tf.transpose(activations, [0, 1, 3, 4, 2])  # (b, 32, 4, 20, 8)
            print('primary activations shape: %s' % activations_transposed.get_shape())

    return activations_transposed, conv
