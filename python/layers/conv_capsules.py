import numpy as np
import tensorflow as tf
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
           pose: (b, 32, 4, 20, 8)
           activation: (b, 14, 14, 32)
    :param kernel_shape: the shape of convolution operation kernel, [kh, kw, i, o] = (3, 3, 32, 32)
    :param strides: often [1, 2, 2, 1] (stride 2), or [1, 1, 1, 1] (stride 1).
    :param routing_ites: number of iterations in EM routing. 3
    :param name: name.

    :return: (poses, activations).

    """

    with tf.variable_scope(name):
        inputs = tf.transpose(inputs, [0, 2, 3, 1, 4])  # (b, 32, 14, 14, 8) -> (b, 14, 14, 32, 8)
        activation_length = inputs.get_shape()[-1].value  # 8

        # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
        # Such that we can later multiple with the transformation matrices to generate the votes.
        inputs_tiled = kernel_tile(inputs, kernel_size, stride)  # (b, 14, 14, 32, 8) -> (b, 6, 6, 3x3=9, 32x8=256)
        print('inputs_tiled shape: %s' % inputs_tiled.get_shape())
        spatial_size_1 = inputs_tiled.get_shape()[1].value
        spatial_size_2 = inputs_tiled.get_shape()[2].value

        inputs_tiled = tf.reshape(
            inputs_tiled,
            [batch_size, spatial_size_1, spatial_size_2, inputs_tiled.get_shape()[3].value, in_capsules,
             activation_length])  # (b, 6, 6, 3x3=9, 32, 8)
        inputs_tiled = tf.transpose(inputs_tiled, [0, 4, 1, 2, 3, 5])  # (b, 32, 6, 6, 3x3=9, 8)

        with tf.variable_scope('votes') as scope:
            # Generate the votes by multiply it with the transformation matrices
            votes = mat_transform(inputs_tiled, out_capsules, kernel_size,
                                  activation_length, batch_size, spatial_size_1,
                                  spatial_size_2)  # (b, 32*9, 32*6*6, 8_out)

            # activations = tf.reduce_sum(votes_reshaped, axis=1)
            print('votes shape: %s' % votes.get_shape())

        with tf.variable_scope('routing'):
            # coupling_coeffs_shape = tf.stack(
            #     [batch_size, in_capsules*kernel_size*kernel_size, out_capsules*spatial_size_1*spatial_size_2])  # (b, 32*9, 32*6*6)
            # # coupling_coeffs = tf.ones(coupling_coeffs_shape)
            # activations, coupling_coeffs = dynamic_routing(
            #     votes=votes,
            #     coupling_coeffs_shape=coupling_coeffs_shape,
            #     num_dims=4,
            #     input_dim=in_capsules*kernel_size*kernel_size,
            #     num_routing=routing_ites,
            #     p=" conv capsules")
            activations = tf.reduce_sum(votes, axis=1)  # (b, 32*6*6, 8_out)
            activations = tf.reshape(activations,
                                     [batch_size, out_capsules, spatial_size_1, spatial_size_2, activation_length])  # (b, 32, 6, 6, 8)
            print('activations shape: %s' % activations.get_shape())

        # return activations, coupling_coeffs
        return activations


def kernel_tile(inputs, kernel, stride):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (?, 14, 14, 32, 8)
    :param kernel: 3
    :param stride: 2

    :return output: (?, 6, 6, 9, 512)
    """

    # (?, 14, 14, 32x(8)=256)
    input_shape = inputs.get_shape()  # (b, 14, 14, 32, 8)
    batch_size = input_shape[0].value
    size = input_shape[4].value
    inputs = tf.reshape(inputs, shape=[-1, input_shape[1].value, input_shape[2].value,
                                       input_shape[3].value * size])  # (b, 14, 14, 32*8)

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
    outputs = tf.reshape(output,
                         shape=[batch_size, outputs_shape[1].value, outputs_shape[2].value, -1, kernel * kernel])
    outputs = tf.transpose(outputs, perm=[0, 1, 2, 4, 3])

    # (b, 6, 6, 9, 32*8)
    return outputs


def mat_transform(inputs, output_cap_size, kernel_size, activation_length, batch_size, spatial_size_1, spatial_size_2):
    """

    :param inputs: (b, 32, 6, 6, 3x3=9, 8)
    :param output_cap_size: 32
    :param activation_length: 8
    :param batch_size: b
    :param spatial_size: 6
    :return:
    """

    inputs_shape = inputs.get_shape()
    input_cap_size = inputs_shape[1].value

    inputs = tf.reshape(inputs,
                        [batch_size, 1, inputs_shape[1].value, spatial_size_1*spatial_size_2,
                         kernel_size*kernel_size, activation_length, 1])  # (b, 1, 32, 6*6, 3x3=9, 8_in, 1)
    inputs = tf.tile(inputs, [1, output_cap_size, 1, 1, 1, 1, activation_length])  # (b, 32, 32, 6*6, 3x3=9, 8_in, 8_out)
    inputs = tf.transpose(inputs, [0, 1, 4, 2, 3, 5, 6])  # (b, 32, 3x3=9, 32, 6*6, 8_in, 8_out)
    inputs = tf.reshape(inputs, [batch_size, input_cap_size*kernel_size*kernel_size,
                                 output_cap_size*spatial_size_1*spatial_size_2, activation_length, activation_length])  # (b, 32*9, 32*6*6, 8_in, 8_out)

    w = tf.contrib.framework.variable(
        'w',
        shape=[1, input_cap_size*kernel_size*kernel_size, output_cap_size, activation_length, activation_length],
        dtype=tf.float32,
        # initializer=tf.truncated_normal_initializer(stddev=5e-5))
        initializer=tf.random_normal_initializer(stddev=0.01))  # (1, 32*9, 32, 8_in, 8_out)

    # reshape to avoid tile rank restriction.
    w = tf.reshape(w, [1, input_cap_size*kernel_size*kernel_size, output_cap_size, 1,
                       activation_length, activation_length])  # (1, 32*9, 32, 1, 8_in, 8_out)
    print('w shape: %s' % w.get_shape())
    w = tf.tile(w, [batch_size, 1, 1, spatial_size_1 * spatial_size_2, 1, 1])  # (b, 32*9, 32, 6*6, 8_int, 8_out)
    w = tf.reshape(w, [batch_size, input_cap_size*kernel_size * kernel_size, output_cap_size*spatial_size_1*spatial_size_2,
                       activation_length, activation_length])  # (b, 32*9, 32*6*6, 8_int, 8_out)

    # inputs = tf.Print(inputs, [inputs])
    multi = inputs * w
    # multi = tf.Print(multi, [multi])
    votes = tf.reduce_sum(multi, axis=4)  # (b, 32*9, 32*6*6, 8_out)
    # votes = tf.Print(votes, [votes])

    return votes  # (b, 32*9, 32*6*6, 8_out)
