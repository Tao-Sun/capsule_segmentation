import tensorflow as tf

from python.layers.convolution import conv2d, deconv
import python.data.hippo.hippo_input as hippo_input


data_input = hippo_input

def inference(inputs, num_classes, training=False, name='unet'):
    with tf.variable_scope(name) as scope:
        conv1 = conv2d(
            inputs,
            kernel=3, out_channels=32, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='relu_conv1'
        )
        print('conv1 shape: %s' % conv1.get_shape())
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
            padding='VALID', data_format='NCHW', name='pool1'
        )
        print('pool1 shape: %s' % pool1.get_shape())

        conv2 = conv2d(
            pool1,
            kernel=3, out_channels=64, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='relu_conv2'
        )
        print('conv2 shape: %s' % conv2.get_shape())
        pool2 = tf.nn.max_pool(
            conv2,
            ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
            padding='VALID', data_format='NCHW', name='pool2'
        )
        print('pool2 shape: %s' % pool2.get_shape())

        conv3 = conv2d(
            pool2,
            kernel=3, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='relu_conv3'
        )
        print('conv3 shape: %s' % conv3.get_shape())

        conv3_dropout = tf.layers.dropout(conv3, 0.5, training=training, name='pool3_dropout')

        deconv1 = deconv(
            conv3_dropout,
            kernel=3, out_channels=128, stride=1, data_format='NCHW',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='deconv1'
        )
        print('deconv1 shape: %s' % deconv1.get_shape())
        deconv1_conv = conv2d(
            deconv1,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='deconv1_conv'
        )
        print('deconv1_conv shape: %s' % deconv1_conv.get_shape())
        concat1 = tf.concat([pool2, deconv1_conv], axis=1, name='concat1')
        dropout1 = tf.layers.dropout(concat1, 0.5, training=training, name='dropout1')
        concat1_conv = conv2d(
            dropout1,
            kernel=2, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='concat1_conv'
        )
        print('concat1_conv shape: %s' % concat1_conv.get_shape())

        deconv2 = deconv(
            concat1_conv,
            kernel=4, out_channels=128, stride=2, data_format='NCHW',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='deconv2'
        )
        print('deconv2 shape: %s' % deconv2.get_shape())
        deconv2_conv = conv2d(
            deconv2,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='deconv2_conv'
        )
        print('deconv2_conv shape: %s' % deconv2_conv.get_shape())
        concat2 = tf.concat([pool1, deconv2_conv], axis=1, name='concat2')
        dropout2 = tf.layers.dropout(concat2, 0.5, training=training, name='dropout2')
        concat2_conv = conv2d(
            dropout2,
            kernel=2, out_channels=128, stride=1, padding='VALID',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='concat2_conv'
        )
        print('concat2_conv shape: %s' % concat2_conv.get_shape())

        deconv3 = deconv(
            concat2_conv,
            kernel=4, out_channels=128, stride=2, data_format='NCHW',
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            name='deconv3'
        )
        print('deconv3 shape: %s' % deconv3.get_shape())
        deconv3_conv = conv2d(
            deconv3,
            kernel=3, out_channels=128, stride=1, padding='SAME',
            activation_fn=tf.nn.relu, name='deconv3_conv'
        )
        print('deconv3_conv shape: %s' % deconv3_conv.get_shape())

        class_conv = conv2d(
            deconv3_conv,
            kernel=3, out_channels=num_classes, stride=1, padding='SAME',
            name='class_conv'
        )
        print('class_conv shape: %s' % class_conv.get_shape())

        label_logits = tf.transpose(class_conv, perm=[0, 2, 3, 1])
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

def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        decay_rate=0.8,
        decay_steps=1000,
        learning_rate=0.001,
        wd=0.0000000001,
        momentum=0.99
    )
