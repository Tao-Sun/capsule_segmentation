import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from python.layers.routing import dynamic_routing


def conv_capsule1d(inputs, kernel_size, stride, routing_ites, in_capsules, out_capsules, batch_size, name):
    """This constructs a convolution capsule layer from a primary or convolution capsule layer.
        i: input capsules (32)
        o: output capsules (32)
        batch size: 24
        spatial dimension: 14x14
        kernel: 3x3
    :param inputs: a primary or convolution capsule layer with poses and activations
           pose: (24, 14, 14, 32, 8)
           activation: (24, 14, 14, 32)
    :param kernel_shape: the shape of convolution operation kernel, [kh, kw, i, o] = (3, 3, 32, 32)
    :param strides: often [1, 2, 2, 1] (stride 2), or [1, 1, 1, 1] (stride 1).
    :param routing_ites: number of iterations in EM routing. 3
    :param name: name.

    :return: (poses, activations).

    """

    with tf.variable_scope(name) as scope:
        activation_length = inputs.get_shape()[-1]  # 8

        # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
        # Such that we can later multiple with the transformation matrices to generate the votes.
        inputs_tiled = kernel_tile(inputs, kernel_size, stride)  # (b, 14, 14, 32, 8) -> (b, 6, 6, 3x3=9, 32x8=256)
        spatial_size = int(inputs_tiled.get_shape()[1])

        inputs_tiled = tf.reshape(
            inputs_tiled,
            [batch_size, spatial_size, spatial_size, inputs_tiled.get_shape()[3], in_capsules, -1,
             1])  # (b, 6, 6, 3x3=9, 32, 8, 1)
        inputs_tiled = tf.transpose(inputs_tiled, [0, 4, 1, 2, 3, 5, 6])  # (b, 32, 6, 6, 3x3=9, 8, 1)

        with tf.variable_scope('votes') as scope:
            # Generate the votes by multiply it with the transformation matrices
            votes = mat_transform(inputs_tiled, out_capsules, kernel_size,
                                  activation_length, batch_size, spatial_size)  # (b, 32, 32, 6, 6, 3x3=9, 8, 1)

            # Reshape the vote for EM routing
            votes_shape = votes.get_shape()
            votes = tf.transpose(votes, [0, 2, 5, 1, 3, 4, 6, 7])  # (b, 32, 9, 32, 6, 6, 8, 1)
            votes_shape = votes.get_shape()
            votes_reshaped = tf.reshape(
                votes,
                shape=[batch_size, votes_shape[1] * votes_shape[2], votes_shape[3] * votes_shape[4] * votes_shape[5],
                       votes_shape[6]]
            )  # (b, 32*9, 32*6*6, 81)
            print('votes_reshaped shape: %s' % votes_reshaped.get_shape())

        with tf.variable_scope('routing') as scope:
            coupling_coeffs_shape = tf.stack(
                [batch_size, in_capsules * kernel_size * kernel_size, out_capsules * spatial_size * spatial_size])
            activations, coupling_coeffs = dynamic_routing(
                votes=votes_reshaped,
                coupling_coeffs_shape=coupling_coeffs_shape,
                num_dims=4,
                input_dim=in_capsules * kernel_size * kernel_size,
                num_routing=routing_ites)
            print('activations shape: %s' % votes_reshaped.get_shape())
            print('coupling_coeffs shape: %s' % coupling_coeffs.get_shape())

        return activations, coupling_coeffs


def kernel_tile(inputs, kernel, stride):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (?, 14, 14, 32, 8)
    :param kernel: 3
    :param stride: 2

    :return output: (?, 6, 6, 9, 512)
    """

    # (?, 14, 14, 32x(8)=256)
    input_shape = inputs.get_shape()  # (b, 14, 14, 32, 8)
    batch_size = input_shape[0]
    size = input_shape[4]
    inputs = tf.reshape(inputs, shape=[-1, input_shape[1], input_shape[2], input_shape[3] * size])  # (b, 14, 14, 32*8)

    inputs_shape = inputs.get_shape()  # (b, 14, 14, 32*8)
    tile_filter = np.zeros(shape=[kernel, kernel, inputs_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0  # (3, 3, 32*8, 9)

    # (3, 3, 32*8, 9)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

    # (b, 6, 6, 32*8*9)
    output = tf.nn.depthwise_conv2d(inputs, tile_filter_op, strides=[
        1, stride, stride, 1], padding='VALID')

    outputs_shape = output.get_shape()  # (b, 6, 6, 32*8*9)
    outputs = tf.reshape(output, shape=[batch_size, outputs_shape[1], outputs_shape[2], -1, kernel * kernel])
    outputs = tf.transpose(outputs, perm=[0, 1, 2, 4, 3])

    # (b, 6, 6, 9, 32*8)
    return outputs


def mat_transform(inputs, output_cap_size, kernel_size, activation_length, batch_size, spatial_size):
    """

    :param inputs: (b, 32, 6, 6, 3x3=9, 8, 1)
    :param output_cap_size: 32
    :param activation_length: 8
    :param batch_size: b
    :param spatial_size: 6
    :return:
    """

    inputs_shape = inputs.get_shape()
    input_cap_size = inputs_shape[1]

    inputs = tf.reshape(inputs_shape[0], 1, inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4],
                        inputs_shape[5], inputs_shape[6])  # (b, 1, 32, 6, 6, 3x3=9, 8, 1)
    inputs = tf.tile(inputs, [1, output_cap_size, 1, 1, 1, 1, 1, 1])  # (b, 32, 32, 6, 6, 3x3=9, 8, 1)

    w = slim.variable(
        'w',
        shape=[1, output_cap_size, input_cap_size, kernel_size * kernel_size, activation_length, activation_length],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))  # (1, 32, 32, 9, 8, 8)
    w_shape = tf.get_shape()
    w = tf.reshape(w, [w_shape[0], w_shape[1], w_shape[2], 1, 1, w_shape[3], w_shape[4], w_shape[5]])
    w = tf.tile(w, [batch_size, 1, 1, spatial_size, spatial_size, 1, 1, 1])  # (b, 32, 32, 6, 6, 9, 8, 8)

    votes = tf.matmul(w, inputs)  # (b, 32, 32, 6, 6, 3x3=9, 8, 1)

    return votes  # (b, 32, 32, 6, 6, 3x3=9, 8, 1)
