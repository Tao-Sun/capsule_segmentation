import tensorflow as tf


def conv2d(inputs, kernel, out_channels, stride, padding, name, data_format='NCHW',
           is_train=True, activation_fn=None, weights_regularizer=None,
           biases_initializer=None, normalizer_fn=None,
           weights_initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)):

    with tf.variable_scope(name) as scope:
        output = tf.contrib.layers.conv2d(
            inputs, out_channels, kernel,
            stride=stride, padding=padding,
            data_format=data_format, scope=scope,
            activation_fn=activation_fn, normalizer_fn=normalizer_fn,
            weights_regularizer=weights_regularizer,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer)

    return output


def deconv(inputs, kernel, out_channels, stride, name, padding='VALID', data_format='NHWC',
           activation_fn=None, weights_regularizer=None, normalizer_fn=None,
           weights_initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)):
    with tf.variable_scope(name) as scope:
        deconv = tf.contrib.layers.conv2d_transpose(
            inputs, num_outputs=out_channels, kernel_size=kernel,
            stride=stride, padding=padding, data_format=data_format,
            scope=scope, activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            weights_regularizer=weights_regularizer,
            weights_initializer=weights_initializer
        )

    return deconv
