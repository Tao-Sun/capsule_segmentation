import tensorflow as tf
import tensorflow.contrib.slim as slim


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


def deconv(labels_encoded, num_classes, name):
    with tf.variable_scope(name):
        deconv1 = tf.contrib.layers.conv2d_transpose(
            labels_encoded,
            num_outputs=num_classes,
            kernel_size=(10, 10),
            stride=2,
            padding='VALID',
            data_format='NHWC',
            scope='deconv1',
            weights_initializer=tf.truncated_normal_initializer(
                stddev=5e-2, dtype=tf.float32),
            activation_fn=tf.nn.relu
        )
        label_logits = tf.contrib.layers.conv2d_transpose(
            deconv1,
            num_outputs=num_classes,
            kernel_size=(9, 9),
            stride=1,
            padding='VALID',
            data_format='NHWC',
            scope='deconv2',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=5e-2, dtype=tf.float32)
        )

    return label_logits
