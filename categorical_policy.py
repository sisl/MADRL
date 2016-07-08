import numpy as np
import tensorflow as tf

import nn
import tfutil
from policy import StochasticPolicy
from distributions import Categorical


class CategoricalMLPPolicy(StochasticPolicy):
    def __init__(self, obsfeat_space, action_space, hidden_spec, tblog, varscope_name):
        self.hidden_spec = hidden_spec
        super(CategoricalMLPPolicy, self).__init__(obsfeat_space, action_space, action_space.n, tblog, varscope_name)
        self._dist = Categorical(self.action_space.n)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.obsfeat_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.n,), initializer=tf.random_uniform_initializer(-.01, .01)) # TODO action_space

        scores_B_Pa = out_layer.output
        actiondist_B_Pa = scores_B_Pa - tfutil.logsumexp(scores_B_Pa, axis=1)
        return actiondist_B_Pa

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        return tfutil.lookup_last_idx(actiondist_B_Pa, input_actions_B_Da[:,0])

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return tfutil.categorical_kl(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa):
        probs_B_A = np.exp(actiondist_B_Pa); assert probs_B_A.shape[1] == self.action_space.n
        return self.distribution.sample(probs_B_A)[:,None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return self.distribution.entropy(np.exp(actiondist_B_Pa))



class PursuitMLPPolicy(CategoricalMLPPolicy):
    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.obsfeat_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, self.action_space.n, initializer=tf.random_uniform_initializer(-.01, .01)) # TODO action_space

        # TODO
        pass
    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        # TODO
        #self._pursuit_look up_last_idx(actiondist_B_Pa, input_actions_B_Da[:,0])
        pass

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        # TODO
        #self._pursuit_categorical_kl()
        pass

    def _sample_from_actiondist(self, actiondist_B_Pa):
        # TODO
        pass

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        # TODO
        pass
