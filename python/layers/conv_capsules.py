import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from python.layers.routing import dynamic_routing


def conv_capsule1d(inputs, kernel_size, stride, routing_ites, in_capsules, out_capsules, activation_length, training, name):
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
        batch_size = inputs.get_shape()[0].value
        i_activation_length = inputs.get_shape()[-1].value  # 8

        # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
        # Such that we can later multiple with the transformation matrices to generate the votes.
        inputs_tiled = kernel_tile(inputs, kernel_size, stride)  # (b, 14, 14, 32, 8) -> (b, 6, 6, 3x3=9, 32x8=256)
        print('inputs_tiled shape: %s' % inputs_tiled.get_shape())
        spatial_size_1 = inputs_tiled.get_shape()[1].value
        spatial_size_2 = inputs_tiled.get_shape()[2].value

        inputs_tiled = tf.reshape(
            inputs_tiled,
            [-1, kernel_size*kernel_size*in_capsules, i_activation_length])  # (b*6*6, 9x32, 8)

        with tf.variable_scope('votes') as scope:
            # Generate the votes by multiply it with the transformation matrices
            votes = mat_transform(inputs_tiled, out_capsules, activation_length)  # (b*6*6, 9*32=288, 32, 8)
            votes_shape = votes.get_shape()
            print('votes shape: %s' % votes_shape)

        with tf.variable_scope('routing'):
            coupling_coeffs_shape = tf.stack(
                [votes_shape[0].value, in_capsules*kernel_size*kernel_size, out_capsules])  # (b*6*6, 32*9=288, 32)
            activations, coupling_coeffs = dynamic_routing(
                votes=votes,
                coupling_coeffs_shape=coupling_coeffs_shape,
                num_dims=4,
                input_dim=in_capsules*kernel_size*kernel_size,
                num_routing=routing_ites,
                caller=" conv capsules")  # (b*6*6, 32, 8), (b*6*6, 32*9, 32)

            activations = tf.reshape(
                activations,
                [-1, activations.get_shape()[-2].value, spatial_size_1, spatial_size_2,
                 activations.get_shape()[-1].value])  # (b, 32, 6, 6, 8)

            print('activations shape: %s' % activations.get_shape())  # (b, 32, 6, 6, 8)
            print('coupling_coeffs shape: %s' % coupling_coeffs.get_shape())  # (b*6*6, 32*9, 32)

        activations = tf.layers.batch_normalization(activations, training=training)
        return activations, coupling_coeffs


def kernel_tile(inputs, kernel, stride):
    """ In each output position, stacks all the input capsules in its receptive field.

    :param inputs: shape (?, 14, 14, 32, 8)
    :param kernel: 3
    :param stride: 2

    :return output: (?, 6, 6, 9, 512)
    """

    # (?, 14, 14, 32x(8)=256)
    inputs_shape = inputs.get_shape()  # (b, 14, 14, 32, 8)
    batch_size = inputs_shape[0].value
    in_caps_dim = inputs_shape[4].value
    inputs = tf.reshape(inputs, shape=[-1, inputs_shape[1].value, inputs_shape[2].value,
                                       inputs_shape[3].value*in_caps_dim])  # (b, 14, 14, 32*8)

    inputs_shape = inputs.get_shape()  # (b, 14, 14, 32*8)
    print('depthwise conv inputs shape in kernel_tile: %s' % inputs_shape)
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
    print('depthwise conv outputs shape in kernel_tile: %s' % outputs_shape)
    outputs = tf.reshape(output,
                         shape=[batch_size, outputs_shape[1].value,
                                outputs_shape[2].value, inputs_shape[3].value, kernel*kernel])
    outputs = tf.transpose(outputs, perm=[0, 1, 2, 4, 3])

    # (b, 6, 6, 9, 32*8)
    return outputs


def mat_transform(inputs, output_cap_size, activation_length):
    """

    :param inputs: (b*6*6, 9*32, 8)
    :param output_cap_size: 32
    :param o_activation_length: 8
    :param size: b*6*6
    :return:
    """

    size = int(inputs.get_shape()[0])  # size=b*6*6
    caps_num_i = int(inputs.get_shape()[1])  # 9*32=288
    i_activation_length = int(inputs.get_shape()[2])
    inputs = tf.reshape(inputs, shape=[size, caps_num_i, 1, 1, i_activation_length])  # (size, 9*32=288, 1, 1, 8)

    w = slim.variable('w', shape=[1, caps_num_i, output_cap_size, i_activation_length, activation_length],
                      dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))  # (1, 288, 32, 8, 8)
    w = tf.tile(w, [size, 1, 1, 1, 1])  # (size, 288, 32, 8, 8)

    inputs = tf.tile(inputs, [1, 1, output_cap_size, 1, 1])  # (size, 288, 32, 1, 8)

    votes = tf.matmul(inputs, w)  # (size, 288, 32, 1, 8)
    votes = tf.reshape(votes, [size, caps_num_i, output_cap_size, activation_length])  # (size, 288, 32, 8)

    # (size, 288, 32, 8)
    return votes
