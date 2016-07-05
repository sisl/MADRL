def categorical_kl(logprobs1_B_A, logprobs2_B_A, name=None):
    """KL divergence between categorical distributions, specified as log probabilities"""
    with tf.op_scope([logprobs1_B_A, logprobs2_B_A], name, 'categorical_kl') as scope:
        kl_B = tf.reduce_sum(tf.exp(logprobs1_B_A) * (logprobs1_B_A - logprobs2_B_A), 1, name=scope)
        return kl_B

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


def lookup_last_idx(a, inds, name=None):
    """
    Looks up indices in a. e.g. a[[1, 2, 3]] = [a[1], a[2], a[3]]
    a is a d1 x d2 ... dn tensor
    inds is a d1 x d2 ... d(n-1) tensor of integers
    returns the tensor
    out[i_1,...,i_{n-1}] = a[i_1,...,i_{n-1}, inds[i_1,...,i_{n-1}]]
    """
    with tf.op_scope([a, inds], name, 'lookup_last_idx') as scope:
        a = tf.convert_to_tensor(a, name='a')
        inds = tf.convert_to_tensor(inds, name='inds')

        # Flatten the arrays
        ashape, indsshape = tf.shape(a), tf.shape(inds)
        aflat, indsflat = tf.reshape(a, [-1]), tf.reshape(inds, [-1])

        # Compute the indices corresponding to inds in the flattened array
        delta = tf.gather(ashape, tf.size(ashape)-1) # i.e. delta = ashape[-1]
        aflatinds = tf.range(0, limit=tf.size(a), delta=delta) + indsflat

        # Look up the desired elements in the flattened array, and reshape
        # to the original shape
        return tf.reshape(tf.gather(aflat, aflatinds), indsshape, name=scope)


def flatcat(arrays, name=None):
    """
    Flattens arrays and concatenates them in order.
    """
    with tf.op_scope(arrays, name, 'flatcat') as scope:
        return tf.concat(0, [tf.reshape(a, [-1]) for a in arrays], name=scope)


def unflatten_into_tensors(flatparams_P, output_shapes, name=None):
    """
    Unflattens a vector produced by flatcat into a list of tensors of the specified shapes.
    """
    with tf.op_scope([flatparams_P], name, 'unflatten_into_tensors') as scope:
        outputs = []
        curr_pos = 0
        for shape in output_shapes:
            size = np.prod(shape)
            flatval = flatparams_P[curr_pos:curr_pos+size]
            outputs.append(tf.reshape(flatval, shape))
            curr_pos += size
            assert curr_pos == flatparams_P.get_shape().num_elements()
        return tf.tuple(outputs, name=scope)

def unflatten_into_vars(flatparams_P, param_vars, name=None):
    """
    Unflattens a vector produced by flatcat into the original variables
    """
    with tf.op_scope([flatparams_P] + param_vars, name, 'unflatten_into_vars') as scope:
        tensors = unflatten_into_tensors(flatparams_P, [v.get_shape().as_list() for v in param_vars])
        return tf.group(*[v.assign(t) for v, t in util.safezip(param_vars, tensors)], name=scope)
