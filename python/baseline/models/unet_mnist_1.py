import tensorflow as tf

from python.layers.convolution import conv2d, deconv

def inference(inputs, num_classes, name='unet'):
    with tf.variable_scope(name) as scope:
        conv1 = conv2d(
            inputs,
            kernel=7, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv1'
        )
        print('conv1 shape: %s' % conv1.get_shape())

        conv11 = conv2d(
            conv1,
            kernel=3, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv11'
        )
        print('conv11 shape: %s' % conv11.get_shape())

        conv12 = conv2d(
            conv11,
            kernel=3, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv12'
        )
        print('conv12 shape: %s' % conv12.get_shape())
        # max_pool1 = tf.nn.max_pool(
        #     conv1,
        #     ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #     padding='VALID', data_format='NCHW', name='max_pool1'
        # )
        conv2 = conv2d(
            conv12,
            kernel=3, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv2'
        )
        print('conv2 shape: %s' % conv2.get_shape())
        # max_pool2 = tf.nn.max_pool(
        #     conv2,
        #     ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #     padding='VALID', data_format='NCHW', name='max_pool2'
        # )
        conv3 = conv2d(
            conv2,
            kernel=5, out_channels=128, stride=2, padding='VALID',
            activation_fn=tf.nn.relu, name='relu_conv3'
        )
        print('conv3 shape: %s' % conv3.get_shape())
        # max_pool3 = tf.nn.max_pool(
        #     conv3,
        #     ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #     padding='VALID', data_format='NCHW', name='max_pool1'
        # )
        conv3 = tf.check_numerics(conv3, message="nan or inf from: conv3")

        deconv1 = deconv(
            conv3,
            kernel=6, out_channels=128, stride=2, data_format='NCHW',
            activation_fn=tf.nn.relu, name='deconv1'
        )
        print('deconv1 shape: %s' % deconv1.get_shape())
        concat1 = tf.concat([conv2, deconv1], axis=1, name='concat1')
        # print('concat1 shape: %s' % concat1.get_shape())
        deconv1_conv = conv2d(
            concat1,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv1_conv'
        )
        deconv2 = deconv(
            deconv1_conv,
            kernel=3, out_channels=128, stride=1, data_format='NCHW',
            activation_fn=tf.nn.relu, name='deconv2'
        )
        print('deconv2 shape: %s' % deconv2.get_shape())
        concat2 = tf.concat([conv12, deconv2], axis=1, name='concat2')
        # print('concat2 shape: %s' % concat2.get_shape())
        deconv2_conv = conv2d(
            concat2,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv2_conv'
        )

        deconv31 = deconv(
            deconv2_conv,
            kernel=3, out_channels=128, stride=1, data_format='NCHW',
            activation_fn=tf.nn.relu, name='deconv31'
        )
        print('deconv31 shape: %s' % deconv31.get_shape())
        concat31 = tf.concat([conv11, deconv31], axis=1, name='concat31')
        # print('concat2 shape: %s' % concat2.get_shape())
        deconv31_conv = conv2d(
            concat31,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv31_conv'
        )

        deconv32 = deconv(
            deconv31_conv,
            kernel=3, out_channels=128, stride=1, data_format='NCHW',
            activation_fn=tf.nn.relu, name='deconv32'
        )
        print('deconv32 shape: %s' % deconv32.get_shape())
        concat32 = tf.concat([conv1, deconv32], axis=1, name='concat31')
        # print('concat2 shape: %s' % concat2.get_shape())
        deconv32_conv = conv2d(
            concat32,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv32_conv'
        )

        deconv3 = deconv(
            deconv32_conv,
            kernel=7, out_channels=num_classes, stride=1, data_format='NCHW',
            activation_fn=tf.nn.relu, name='deconv3'
        )
        # print('deconv3 shape: %s' % deconv3.get_shape())
        deconv3_conv = conv2d(
            deconv3,
            kernel=3, out_channels=num_classes, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv3_conv'
        )

        label_logits = tf.transpose(deconv3_conv, perm=[0, 2, 3, 1])
        label_logits = tf.check_numerics(label_logits, message="nan or inf from: label_logits")
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
        balanced_decode_loss = batch_decode_loss

        tf.add_to_collection('losses', balanced_decode_loss)
        tf.summary.scalar('decode_loss', balanced_decode_loss)
