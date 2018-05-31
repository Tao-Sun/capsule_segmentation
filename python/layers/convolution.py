import tensorflow as tf


def conv2d(inputs, kernel, out_channels, stride, padding, name, data_format='NCHW', is_train=True, activation_fn=None):
    # with slim.arg_scope([slim.conv2d], trainable=is_train):
    #     with tf.variable_scope(name) as scope:
    #         output = slim.conv2d(inputs,
    #                              out_channels, kernel,
    #                              stride=stride, padding=padding, data_format=data_format,
    #                              scope=scope, activation_fn=activation_fn)
    # tf.logging.info(f"{name} output shape: {output.get_shape()}")

    with tf.variable_scope(name) as scope:
        output = tf.contrib.layers.conv2d(
            inputs, out_channels, kernel,
            stride=stride, padding=padding, data_format=data_format,
            scope=scope, activation_fn=activation_fn,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=5e-2, dtype=tf.float32))

    return output


def deconv(inputs, kernel, out_channels, stride, name, padding='VALID', data_format='NHWC', activation_fn=None):
    with tf.variable_scope(name) as scope:
        deconv = tf.contrib.layers.conv2d_transpose(
            inputs, num_outputs=out_channels, kernel_size=kernel,
            stride=stride, padding=padding, data_format=data_format,
            scope=scope, activation_fn=activation_fn,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=5e-2, dtype=tf.float32)
        )

    return deconv
