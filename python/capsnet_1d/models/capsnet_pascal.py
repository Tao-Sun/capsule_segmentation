import tensorflow as tf

from python.layers.convolution import conv2d, deconv
from python.layers.primary_capsules import primary_caps1d
from python.layers.class_capsules import class_caps1d

def inference(inputs, num_classes, training=True, routing_ites=3, name='unet_pascal'):
    with tf.variable_scope(name) as scope:
        inputs_shape = inputs.get_shape()
        batch_size = inputs_shape[0].value

        combine_conv1 = _combine_conv(inputs, '1', training, out_channels=16)
        combine_conv2 = _combine_conv(combine_conv1, '2', training, out_channels=32)
        combine_conv3 = _combine_conv(combine_conv2, '3', training, out_channels=64)
        combine_conv4 = _combine_conv(combine_conv3, '4', training, out_channels=128)

        print("\nprimary layer:")
        primary_out_capsules = 32
        primary_caps_activations, primary_conv = primary_caps1d(
            combine_conv4,
            kernel_size=5, out_capsules=primary_out_capsules, stride=2,
            padding='VALID', activation_length=8, name='primary_caps'
        )
        print('primary_conv shape: %s' % primary_conv.get_shape())

        print("\nclass capsule layer:")
        class_caps_activations, class_coupling_coeffs = class_caps1d(
            primary_caps_activations,
            num_classes=num_classes, activation_length=16, routing_ites=routing_ites,
            batch_size=batch_size, name='class_capsules')
        print('class_coupling_coeffs shape: %s' % class_coupling_coeffs.get_shape())
        print('class_caps_activations shape: %s' % class_caps_activations.get_shape())


        capsule_probs = tf.norm(primary_caps_activations, axis=-1)
        caps_probs_tiled = tf.tile(tf.expand_dims(capsule_probs, -1), [1, 1, 1, 1, num_classes])

        primary_activations_shape = primary_caps_activations.get_shape()
        height, width = primary_activations_shape[2].value, primary_activations_shape[3].value
        coupling_coeff_reshaped = tf.reshape(class_coupling_coeffs,
                                             [batch_size, primary_out_capsules, height, width, num_classes])

        primary_labels = tf.reduce_sum(coupling_coeff_reshaped * caps_probs_tiled, 1)
        print('\nprimary_labels shape: %s\n' % primary_labels.get_shape())
        concat1 = tf.concat([primary_conv, tf.transpose(primary_labels, perm=[0, 3, 1, 2])], axis=1, name='concat1')
        primary_label_conv = conv2d(
            concat1,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, data_format='NCHW', name='primary_label_conv'
        )

        combine_deconv5 = _combine_deconv(primary_label_conv, '5', training, combine_conv4,
                                          deconv_out_channels=256)
        combine_deconv4 = _combine_deconv(combine_deconv5, '4', training, combine_conv3,
                                          kernel_deconv=[7, 7], deconv_out_channels=128)
        combine_deconv3 = _combine_deconv(combine_deconv4, '3', training, combine_conv2,
                                          kernel_deconv=[6, 6], deconv_out_channels=64)
        combine_deconv2 = _combine_deconv(combine_deconv3, '2', training, combine_conv1,
                                          kernel_deconv=[6, 7], deconv_out_channels=64)
        combine_deconv1 = _combine_deconv(combine_deconv2, '1', training, None,
                                          kernel_deconv=[6, 7], deconv_out_channels=32)

        conv = conv2d(
            combine_deconv1,
            kernel=3, out_channels=num_classes, stride=1,
            padding='SAME', name='label_conv'
        )

        label_logits = tf.transpose(conv, perm=[0, 2, 3, 1])
        # label_logits = tf.check_numerics(label_logits, message="nan or inf from: label_logits")
        print('label_logits shape: %s' % label_logits.get_shape())
        return class_caps_activations, None, label_logits


def _combine_conv(inputs, layer, training, kernel=5, padding='VALID', out_channels=32):
    conv = conv2d(
        inputs,
        kernel=kernel, out_channels=out_channels, stride=1,
        padding=padding, name='combine_conv' + layer
    )
    print('conv layer: %s, shape: %s' % (layer, conv.get_shape()))
    bn = tf.layers.batch_normalization(conv, axis=1, center=True, scale=False,
                                       training=training, name='bn' + layer)
    relu = tf.nn.relu(bn, name='relu' + layer)
    pool = tf.layers.max_pooling2d(inputs=relu, pool_size=[2, 2], strides=[2, 2],
                                   data_format="channels_first", name='pool' + layer)

    print('layer: %s, shape: %s\n' % (layer, pool.get_shape()))
    return pool


def _combine_deconv(inputs, layer, training, conv_val, kernel_deconv=5, kernel_conv=3,
                    deconv_out_channels=32, conv_out_channels=32, conv_padding='SAME'):
    deconv1 = deconv(
        inputs,
        kernel=kernel_deconv, out_channels=deconv_out_channels, stride=2,
        data_format='NCHW', activation_fn=tf.nn.relu, name='deconv' + layer
    )
    print('deconv layer: %s, deconv shape: %s' % (layer, deconv1.get_shape()))

    if conv_val is not None:
        print('conv layer to be concatenated, shape: %s' % (conv_val.get_shape()))
        concat1 = tf.concat([conv_val, deconv1],
                            axis=1, name='concat' + layer)
    else:
        concat1 = deconv1
    conv = conv2d(
        concat1,
        kernel=kernel_conv, out_channels=conv_out_channels, stride=1, data_format='NCHW',
        padding=conv_padding, name='deconv_conv' + layer
    )
    print('deconv layer: %s, conv shape: %s' % (layer, conv.get_shape()))
    bn = tf.layers.batch_normalization(conv, axis=-1, center=True, scale=False,
                                       training=training, name='deconv_bn' + layer)
    print('deconv layer: %s, final shape: %s\n' % (layer, bn.get_shape()))
    return bn
