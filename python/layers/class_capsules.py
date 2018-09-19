import tensorflow as tf

from python.layers.routing import dynamic_routing


def class_caps1d(inputs, num_classes, activation_length, routing_ites, batch_size, name):
    """

    :param inputs: (b, 32, 8, 6, 6)
    :param num_classes:
    :param activation_length:
    :param routing_ites:
    :param batch_size:
    :param name:
    :return:
    """

    inputs = tf.check_numerics(inputs, message="nan or inf from: inputs in class capsules")
    inputs_shape = inputs.get_shape()  # (b, 32, 4, 20, 8)
    in_capsules = inputs_shape[1].value
    in_height, in_width = inputs_shape[2].value, inputs_shape[3].value
    in_pose_length = inputs_shape[4].value

    inputs_3d = tf.reshape(inputs, [batch_size, -1, in_pose_length])  # (b, 32*4*20, 8)

    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(
                    'weights_1',
                    [in_capsules, in_pose_length, num_classes*activation_length],
                    initializer=tf.truncated_normal_initializer(
                        stddev=5e-2, dtype=tf.float32),
                    dtype=tf.float32)  # (32*4*20, 8, 2*64)
                weights_titled = tf.tile(tf.expand_dims(weights, 1), [1, in_height*in_width, 1, 1])
                weights_reshaped = \
                    tf.reshape(weights_titled, [in_capsules*in_height*in_width, in_pose_length, num_classes*activation_length])
                print('weights_reshaped shape: %s' % weights_reshaped.get_shape())


        with tf.name_scope('Wx_plus_b'):
            input_tiled = tf.tile(tf.expand_dims(inputs_3d, -1), [1, 1, 1, num_classes * activation_length])  # (b, 32*4*20, 8, 2*64)
            print('input_tiled shape: %s' % input_tiled.get_shape())
            votes = tf.reduce_sum(input_tiled * weights_reshaped, axis=2)
            votes_reshaped = tf.reshape(votes, [-1, in_capsules * in_height * in_width, num_classes, activation_length])  # (b, 32*4*20, 2, 64)
            votes_reshaped = tf.check_numerics(votes_reshaped, message="nan or inf from: votes_reshaped in class capsules")
            print('votes_reshaped shape: %s' % votes_reshaped.get_shape())

        with tf.name_scope('routing'):
            coupling_coeffs_shape = tf.stack([batch_size, in_capsules * in_height * in_width, num_classes])  # (b, 32*4*20, 2)
            activations, coupling_coeffs = dynamic_routing(
                votes=votes_reshaped,
                coupling_coeffs_shape=coupling_coeffs_shape,
                num_dims=4,
                input_dim=in_capsules * in_height * in_width,
                num_routing=routing_ites,
                caller=" class capsules")
            # activations = tf.Print(activations, [tf.constant("class_caps_activations"), activations])
            activations = tf.check_numerics(activations, message="nan or inf from: activations in class capsules")
            coupling_coeffs = tf.check_numerics(coupling_coeffs, message="nan or inf from: coupling_coeffs in class capsules")
            print('class capsule activations shape: %s' % votes_reshaped.get_shape())
            print('class capsuel coupling_coeffs shape: %s' % coupling_coeffs.get_shape())

    return activations, coupling_coeffs
