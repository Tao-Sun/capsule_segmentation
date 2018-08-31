import tensorflow as tf
import numpy as np

from python.layers.convolution import conv2d, deconv


vgg_weights = np.load("./python/data/pascal/vgg16_weights.npz")
keys = sorted(vgg_weights.keys())
init_weights = {}
for i, key in enumerate(keys):
    # print i, key, np.shape(vgg_weights[key])
    init_weights[key] = vgg_weights[key]
init_weights['conv6_W'] = vgg_weights['fc6_W']
init_weights['conv6_b'] = vgg_weights['fc6_b']
init_weights['conv7_W'] = vgg_weights['fc7_W']
init_weights['conv7_b'] = vgg_weights['fc7_b']


def inference(inputs, num_classes, feature_scale=1, training=True, name='unet'):
    with tf.variable_scope(name) as scope:
        filters = [64, 128, 256, 512, 512]
        filters = [int(x / feature_scale) for x in filters]

        padded_input = tf.pad(inputs, tf.constant([[0, 0], [0, 0], [100, 100], [100, 100]]), "CONSTANT")

        combine_conv1_1 = _combine_conv(padded_input, '1_1', training, in_channels=3, out_channels=filters[0])
        combine_conv1_2 = _combine_conv(combine_conv1_1, '1_2', training,
                                        in_channels=filters[0], out_channels=filters[0])
        pool1 = tf.layers.max_pooling2d(inputs=combine_conv1_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool1')
        print('pool1 shape: %s\n' % (pool1.get_shape()))
        combine_conv2_1 = _combine_conv(pool1, '2_1', training, in_channels=filters[0], out_channels=filters[1])
        combine_conv2_2 = _combine_conv(combine_conv2_1, '2_2', training,
                                        in_channels=filters[1], out_channels=filters[1])
        pool2 = tf.layers.max_pooling2d(inputs=combine_conv2_2, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool2')
        print('pool2 shape: %s\n' % (pool2.get_shape()))
        combine_conv3_1 = _combine_conv(pool2, '3_1', training,
                                        in_channels=filters[1], out_channels=filters[2])
        combine_conv3_2 = _combine_conv(combine_conv3_1, '3_2', training,
                                        in_channels=filters[2], out_channels=filters[2])
        combine_conv3_3 = _combine_conv(combine_conv3_2, '3_3', training,
                                        in_channels=filters[2], out_channels=filters[2])
        pool3 = tf.layers.max_pooling2d(inputs=combine_conv3_3, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool3')
        print('pool3 shape: %s\n' % (pool3.get_shape()))
        combine_conv4_1 = _combine_conv(pool3, '4_1', training, in_channels=filters[2], out_channels=filters[3])
        combine_conv4_2 = _combine_conv(combine_conv4_1, '4_2', training,
                                        in_channels=filters[3], out_channels=filters[3])
        combine_conv4_3 = _combine_conv(combine_conv4_2, '4_3', training,
                                        in_channels=filters[3], out_channels=filters[3])
        pool4 = tf.layers.max_pooling2d(inputs=combine_conv4_3, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool4')
        print('pool4 shape: %s\n' % (pool4.get_shape()))
        combine_conv5_1 = _combine_conv(pool4, '5_1', training, in_channels=filters[3], out_channels=filters[4])
        combine_conv5_2 = _combine_conv(combine_conv5_1, '5_2', training,
                                        in_channels=filters[4], out_channels=filters[4])
        combine_conv5_3 = _combine_conv(combine_conv5_2, '5_3', training,
                                        in_channels=filters[4], out_channels=filters[4])
        pool5 = tf.layers.max_pooling2d(inputs=combine_conv5_3, pool_size=[2, 2], strides=[2, 2],
                                        data_format="channels_first", name='pool5')
        print('pool5 shape: %s\n' % (pool5.get_shape()))

        combine_conv6 = _combine_conv(pool5, '6', training, kernel=7,
                                      in_channels=filters[4], out_channels=4096, padding='VALID')
        drop6 = tf.nn.dropout(combine_conv6, 0.5)
        combine_conv7 = _combine_conv(drop6, '7', training, kernel=1,
                                      in_channels=4096, out_channels=4096, padding='VALID')
        drop7 = tf.nn.dropout(combine_conv7, 0.5)
        score_fr = conv2d(
            drop7, kernel=1,
            out_channels=num_classes, stride=1,
            padding='VALID', name='score_fr',
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005)
        )
        kernel_upscore2 = 4
        upscore2 = deconv(
            score_fr,
            kernel=kernel_upscore2, out_channels=num_classes, stride=2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
            weights_initializer=_deconv_intializer(kernel_upscore2),
            data_format='NCHW', activation_fn=None, name='upscore2'
        )
        print('upscore2 shape: %s\n' % (upscore2.get_shape()))

        score_pool4 = conv2d(
            pool4, kernel=1,
            out_channels=num_classes, stride=1,
            padding='VALID', name='score_pool4',
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005)
        )
        print('score_pool4 shape: %s' % (score_pool4.get_shape()))
        score_pool4c = _crop(score_pool4, upscore2)  # need crop here
        print('score_pool4c shape: %s' % (score_pool4c.get_shape()))
        fuse_pool4 = upscore2 + score_pool4c
        kernel_upscore_pool4 = 4
        upscore_pool4 = deconv(
            fuse_pool4,
            kernel=kernel_upscore_pool4, out_channels=num_classes, stride=2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
            weights_initializer=_deconv_intializer(kernel_upscore_pool4),
            data_format='NCHW', activation_fn=None, name='upscore_pool4'
        )
        print('upscore_pool4 shape: %s\n' % (upscore_pool4.get_shape()))

        score_pool3 = conv2d(
            pool3, kernel=1,
            out_channels=num_classes, stride=1,
            padding='VALID', name='score_pool3',
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005)
        )
        score_pool3c = _crop(score_pool3, upscore_pool4)  # need crop here
        print('score_pool3c shape: %s' % (score_pool3c.get_shape()))
        fuse_pool3 = upscore_pool4 + score_pool3c
        kernel_upscore_pool8 = 16
        upscore_pool8 = deconv(
            fuse_pool3,
            kernel=kernel_upscore_pool8, out_channels=num_classes, stride=8,
            data_format='NCHW', activation_fn=None, name='upscore_pool8',
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
            weights_initializer=_deconv_intializer(kernel_upscore_pool8)
        )
        print('upscore_pool8 shape: %s\n' % (upscore_pool8.get_shape()))

        score = _crop(upscore_pool8, inputs)  # need crop here
        print('score shape: %s\n' % (score.get_shape()))
        label_logits = tf.transpose(score, perm=[0, 2, 3, 1])
        print('label_logits shape: %s' % label_logits.get_shape())
        return label_logits


def loss(labels2d, label_logits, num_classes):
    with tf.name_scope('loss'):
        one_hot_labels = tf.one_hot(labels2d, depth=num_classes)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,
        #                                                         logits=label_logits)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels2d,
                                                               logits=label_logits)

        class_weights = tf.constant([1.0] * num_classes)
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=3)

        weighted_losses = cross_entropy * weights
        print('labels2d shape: %s' % labels2d.get_shape())
        print('label_logits shape: %s' % label_logits.get_shape())
        print('cross_entropy shape: %s' % weighted_losses.get_shape())

        batch_decode_loss = tf.reduce_sum(cross_entropy)
        balanced_decode_loss = batch_decode_loss

        tf.add_to_collection('losses', balanced_decode_loss)
        tf.summary.scalar('decode_loss', balanced_decode_loss)


def _combine_conv(inputs, layer, training, in_channels, out_channels, kernel=3, padding='SAME', is_batchnorm=False):

    def _weights_initializer(weights):
        if weights.shape != (kernel, kernel, in_channels, out_channels):
            weights = np.reshape(weights, (kernel, kernel, in_channels, out_channels))
        return tf.constant_initializer(weights)

    conv = conv2d(
        inputs, kernel=kernel,
        out_channels=out_channels, stride=1,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
        weights_initializer=_weights_initializer(init_weights['conv' + layer + '_W']),
        biases_initializer=tf.constant_initializer(init_weights['conv' + layer + '_b']),
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


def _deconv_intializer(kernel):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (kernel + 1) // 2
    if kernel % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel, :kernel]

    bilinear_filter = (1 - abs(og[0] - center) / factor) * \
                      (1 - abs(og[1] - center) / factor)
    return tf.constant_initializer(bilinear_filter)


def _crop(t1, t2):
    t1_shape = t1.get_shape()
    t2_shape = t2.get_shape()
    # offsets for the top left corner of the crop
    offsets = [0, 0, int(t1_shape[2] - t2_shape[2]) // 2, int(t1_shape[3] - t2_shape[3]) // 2]
    size = [-1, -1, int(t2_shape[2]), int(t2_shape[3])]
    t1_crop = tf.slice(t1, offsets, size)
    return t1_crop
