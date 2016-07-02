import tensorflow as tf

import tfutil
from .policy import StochasticPolicy

class CategoricalMLPPolicy(StochasticPolicy):
    def __init__(self, env_spec, cfg):
        self.cfg = cfg
        super(CategoricalMLPPolicy, self).__init__(env_spec)

    def _make_actiondist_computation(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.size,), initializer=tf.random_uniform_initializer(-.01, .01)) # TODO action_space

        scores_B_Da = out_layer.output
        actiondist_B_Pa = scores_B_Pa - tfutil.logsumexp(scores_B_Pa, axis=1)
        return actiondist_B_Pa

    def _make_actiondist_logprob_computation(self, actiondist_B_Pa, input_actions_B_Da):
        # TODO
        pass

    def _make_actiondist_kl_computation(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        # TODO
        pass

    def _sample_from_actiondist(self, actiondist_B_Pa):
        # TODO
        pass

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        # TODO
        pass


class PursuitMLPPolicy(CategoricalMLPPolicy):
    def _make_actiondist_computation(self, obsfeat_B_Df):
        # TODO
        pass
    def _make_actiondist_logprob_computation(self, actiondist_B_Pa, input_actions_B_Da):
        # TODO
        pass

    def _make_actiondist_kl_computation(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        # TODO
        pass

    def _sample_from_actiondist(self, actiondist_B_Pa):
        # TODO
        pass

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        # TODO
        pass
