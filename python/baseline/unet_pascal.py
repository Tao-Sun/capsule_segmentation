import tensorflow as tf

from python.layers.convolution import conv2d, deconv

def inference(inputs, num_classes, feature_scale=2, training=True, name='unet'):
    with tf.variable_scope(name) as scope:
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        combine_conv1_1 = _combine_conv(inputs, '1_1', training, out_channels=filters[0])
        combine_conv1_2 = _combine_conv(combine_conv1_1, '1_2', training, out_channels=filters[0])
        pool1 = tf.layers.max_pooling2d(inputs=combine_conv1_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool1')
        print('pool1 shape: %s\n' % (pool1.get_shape()))
        combine_conv2_1 = _combine_conv(pool1, '2_1', training, out_channels=filters[1])
        combine_conv2_2 = _combine_conv(combine_conv2_1, '2_2', training, out_channels=filters[1])
        pool2 = tf.layers.max_pooling2d(inputs=combine_conv2_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool2')
        print('pool2 shape: %s\n' % (pool2.get_shape()))
        combine_conv3_1 = _combine_conv(pool2, '3_1', training, out_channels=filters[2])
        combine_conv3_2 = _combine_conv(combine_conv3_1, '3_2', training, out_channels=filters[2])
        pool3 = tf.layers.max_pooling2d(inputs=combine_conv3_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool3')
        print('pool3 shape: %s\n' % (pool3.get_shape()))
        combine_conv4_1 = _combine_conv(pool3, '4_1', training, out_channels=filters[3])
        combine_conv4_2 = _combine_conv(combine_conv4_1, '4_2', training, out_channels=filters[3])
        pool4 = tf.layers.max_pooling2d(inputs=combine_conv4_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool4')
        pool4 = tf.nn.dropout(pool4, 0.5)
        print('pool4 shape: %s\n' % (pool4.get_shape()))

        center = _combine_conv(pool4, '5', training, out_channels=filters[4])

        combine_deconv4 = _combine_deconv(center, '4', training, combine_conv4_2,
                                          deconv_out_channels=filters[3])
        combine_deconv3 = _combine_deconv(combine_deconv4, '3', training, combine_conv3_2,
                                          deconv_out_channels=filters[2])
        combine_deconv2 = _combine_deconv(combine_deconv3, '2', training, combine_conv2_2,
                                          deconv_out_channels=filters[1])
        combine_deconv1 = _combine_deconv(combine_deconv2, '1', training, combine_conv1_2,
                                          deconv_out_channels=filters[0])\

        final_deconv = deconv(
            combine_deconv1,
            kernel=5, out_channels=num_classes, stride=1,
            data_format='NCHW', name='final_deconv'
        )
        final = conv2d(
            final_deconv,
            kernel=1, out_channels=num_classes, stride=1,
            padding='SAME', name='final_conv'
        )

        label_logits = tf.transpose(final, perm=[0, 2, 3, 1])
        print('label_logits shape: %s' % label_logits.get_shape())
        return label_logits


def loss(labels2d, label_logits, num_classes):
    with tf.name_scope('loss'):
        one_hot_labels = tf.one_hot(labels2d, depth=num_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,
                                                                logits=label_logits)

        class_weights = tf.constant([1.0] + [5.0] * (num_classes - 1))
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=3)

        weighted_losses = cross_entropy * weights
        print('labels2d shape: %s' % labels2d.get_shape())
        print('label_logits shape: %s' % label_logits.get_shape())
        print('cross_entropy shape: %s' % weighted_losses.get_shape())

        batch_decode_loss = tf.reduce_mean(weighted_losses)
        balanced_decode_loss = 5 * batch_decode_loss

        tf.add_to_collection('losses', balanced_decode_loss)
        tf.summary.scalar('decode_loss', balanced_decode_loss)

def _combine_conv(inputs, layer, training, out_channels, padding='VALID', is_batchnorm=True):
    conv = conv2d(
        inputs, kernel=3,
        out_channels=out_channels, stride=1,
        padding=padding, name='combine_conv' + layer
    )
    print('conv layer: %s, shape: %s' % (layer, conv.get_shape()))
    if is_batchnorm:
        bn = tf.layers.batch_normalization(conv, axis=1, center=True, scale=False,
                                           training=training, name='bn' + layer)
    else:
        bn = conv
    relu = tf.nn.relu(bn, name='relu' + layer)

    print('layer: %s, shape: %s' % (layer, relu.get_shape()))
    return relu


def _combine_deconv(inputs, layer, training, conv_concat, deconv_out_channels):
    deconv1 = deconv(
        inputs,
        kernel=2, out_channels=deconv_out_channels, stride=2,
        data_format='NCHW', activation_fn=tf.nn.relu, name='deconv' + layer
    )
    print('deconv layer: %s, deconv shape: %s' % (layer, deconv1.get_shape()))

    if conv_concat is not None:
        print('conv layer to be concatenated, shape: %s' % (conv_concat.get_shape()))
        offset1 = int(conv_concat.get_shape()[2] - deconv1.get_shape()[2])
        padding1 = [offset1 // 2, offset1 - offset1 // 2]
        offset2 = int(conv_concat.get_shape()[3] - deconv1.get_shape()[3])
        padding2 = [offset2 // 2, offset2 - offset2 // 2]
        padding = tf.constant([[0, 0], [0, 0], padding1, padding2])

        deconv1 = tf.pad(deconv1, padding)
        print('deconv layer shape to be concatenated: %s' % (deconv1.get_shape()))
        concat = tf.concat([conv_concat, deconv1],
                            axis=1, name='concat' + layer)
        concat = tf.nn.dropout(concat, 0.5)
    else:
        concat = deconv1

    deconv_conv_1 = _combine_conv(concat, 'deconv_' + layer + '_conv1', training, padding='SAME',
                                  is_batchnorm=False, out_channels=deconv_out_channels)
    deconv_conv_2 = _combine_conv(deconv_conv_1, 'deconv_' + layer + '_conv2', training, padding='SAME',
                                  is_batchnorm=False, out_channels=deconv_out_channels)
    print('deconv layer: %s, final shape: %s\n' % (layer, deconv_conv_2.get_shape()))
    return deconv_conv_2
