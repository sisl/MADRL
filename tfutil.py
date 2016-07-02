def logsumexp(a, axis, name=None):
    """
    Like scipy.misc.logsumexp with keepdims=True
    (does NOT eliminate the singleton axis)
    """
    with tf.op_scope([a, axis], name, 'logsumexp') as scope:
        a = tf.convert_to_tensor(a, name='a')
        axis = tf.convert_to_tensor(axis, name='axis')

        amax = tf.reduce_max(a, axis, keep_dims=True)
        shifted_result = tf.log(tf.reduce_sum(tf.exp(a - amax), axis, keep_dims=True))
        return tf.add(shifted_result, amax, name=scope)
