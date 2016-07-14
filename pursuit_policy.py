import numpy as np
import tensorflow as tf

from rltools import nn
from rltools import tfutil
from rltools.policy import StochasticPolicy
from rltools.distributions import Distribution


class FactoredCategorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        self._dim

    def entropy(self, probs_N_H_K):
        tmp = -probs_N_H_K * np.log(probs_N_H_K)
        tmp[~np.isfinite(tmp)] = 0
        return tmp.sum(axis=2)

    def sample(self, probs_N_H_K):
        """Sample from N factored categorical distributions"""
        N, H, K = probs_N_H_K.shape
        return np.array([np.random.choice(K, p=probs_N_H_K[i,j,:]) for i in xrange(N) for j in xrange(H)])

    def kl_expr(self, logprobs1_B_N_A, logprobs2_B_N_A, name=None):
        """KL divergence between facotored categorical distributions"""
        with tf.op_scope([logprobs1_B_N_A, logprobs2_B_N_A], name, 'fac_categorical_kl') as scope:
            kl_B = tf.reduce_sum(tf.reduce_sum(tf.exp(logprobs1_B_N_A) * (logprobs1_B_N_A - logprobs2_B_N_A), 2), 1, name=scope)
        return kl_B


class PursuitCentralMLPPolicy(StochasticPolicy):
    def __init__(self, obsfeat_space, action_space, n_agents,
                 hidden_spec, enable_obsnorm, tblog, varscope_name):
        self.hidden_spec = hidden_spec
        self._n_agents = n_agents
        self._dist = FactoredCategorical(action_space.n)
        super(PursuitCentralMLPPolicy, self).__init__(obsfeat_space, action_space,
                                                      action_space.n, enable_obsnorm,
                                                      tblog, varscope_name)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.obsfeat_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape,
                                       (self.action_space.n,),
                                       initializer=tf.zeros_initializer)

        scores_B_NPa = out_layer.output
        scores_B_N_Pa = tf.reshape(scores_B_NPa, (-1, self._n_agents, self.action_space.n/self._n_agents))
        actiondist_B_N_Pa = scores_B_N_Pa - tfutil.logsumexp(scores_B_N_Pa, axis=2)
        actiondist_B_NPa = tf.reshape(actiondist_B_N_Pa, (-1, self.action_space.n))
        return actiondist_B_NPa

    def _make_actiondist_logprobs_ops(self, actiondist_B_NPa, input_actions_B_NDa):
        actiondist_B_N_Pa = tf.reshape(actiondist_B_NPa, (-1, self._n_agents, self.action_space.n))
        input_actions_B_N_Da = tf.reshape(input_actions_B_NDa, (-1, self._n_agents, self.action_space.n/self._n_agents))
        return tfutil.lookup_last_idx(actiondist_B_N_Pa, input_actions_B_N_Da[:,:,0]) # FIXME?

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_NPa, actiondist_B_NPa):
        proposal_actiondist_B_N_Pa = tf.reshape(proposal_actiondist_B_NPa , (-1, self._n_agents, self.action_space.n/self._n_agents))
        actiondist_B_N_Pa = tf.reshape(actiondist_B_NPa, (-1, self._n_agents, self.action_space.n/self._n_agents))
        return self.distribution.kl_expr(proposal_actiondist_B_N_Pa, actiondist_B_N_Pa)

    def _sample_from_actiondist(self, actiondist_B_NPa, deterministic=False):
        actiondist_B_N_Pa = np.reshape(actiondist_B_NPa, (-1, self._n_agents, self.action_space.n/self._n_agents))
        probs_B_N_A = np.exp(actiondist_B_N_Pa); assert probs_B_N_A.shape[2] == self.action_space.n/self._n_agents
        if deterministic:
            action_B_N = np.argmax(probs_B_N_A, axis=2)
        else:
            action_B_N = self.distribution.sample(probs_B_N_A)

        return action_B_N

    def _compute_actiondist_entropy(self, actiondist_B_NPa):
        actiondist_B_N_Pa = actiondist_B_NPa.reshape((-1, self._n_agents, self.action_space.n/self._n_agents))
        return self.distribution.entropy(np.exp(actiondist_B_N_Pa))
