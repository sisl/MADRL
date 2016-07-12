import numpy as np
import tensorflow as tf


class Distribution(object):

    @property
    def dim(self):
        raise NotImplementedError()

    def kl(self, old, new):
        raise NotImplementedError()

    def entropy(self, probs_N_K):
        raise NotImplementedError()

    def sample(self, probs_N_K):
        raise NotImplementedError()


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def entropy(self, probs_N_K):
        tmp = -probs_N_K * np.log(probs_N_K + 1e-10)
        tmp[~np.isfinite(tmp)] = 0
        return tmp.sum(axis=1)

    def sample(self, probs_N_K):
        """Sample from N categorical distributions, each over K outcomes"""
        N, K = probs_N_K.shape
        return np.array([np.random.choice(K, p=probs_N_K[i,:]) for i in xrange(N)])

    def kl_expr(self, logprobs1_B_A, logprobs2_B_A, name=None):
        """KL divergence between categorical distributions, specified as log probabilities"""
        with tf.op_scope([logprobs1_B_A, logprobs2_B_A], name, 'categorical_kl') as scope:
            kl_B = tf.reduce_sum(tf.exp(logprobs1_B_A) * (logprobs1_B_A - logprobs2_B_A), 1, name=scope)
        return kl_B
