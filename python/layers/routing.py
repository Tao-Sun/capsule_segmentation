import tensorflow as tf
import numpy as np



def dynamic_routing(votes, coupling_coeffs_shape, num_dims, input_dim, num_routing):
    """

    :param votes:
    :param coupling_coeffs_shape:
    :param num_dims:
    :param input_dim:
    :param num_routing:
    :return:
    """

    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
        votes_t_shape += [i + 4]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
        r_t_shape += [i + 4]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def squash(input_tensor):
        """Applies norm nonlinearity (squash) to a capsule layer.

        Args:
          input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
            fully connected capsule layer or
            [batch, num_channels, num_atoms, height, width] for a convolutional
            capsule layer.

        Returns:
          A tensor with same shape as input (rank 3) for output of this layer.
        """
        with tf.name_scope('norm_non_linearity'):
            norm = tf.norm(input_tensor, axis=2, keep_dims=True)
            norm_squared = norm * norm
            return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.log_softmax(logits, dim=2)
        # votes_trans_print = tf.Print(votes_trans, [votes_trans, route])
        # preactivate_unrolled = route * votes_trans_print
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1)

        activation = squash(preactivate)
        activations = activations.write(i, activation)

        # distances: [batch, input_dim, output_dim]
        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)

        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    coupling_coeffs = tf.fill(coupling_coeffs_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    print('coupling_coeffs shape: %s' % coupling_coeffs.get_shape())
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, coupling_coeffs, activations],
        swap_memory=True)

    return activations.read(num_routing - 1), logits