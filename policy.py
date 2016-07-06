from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

import nn

class Policy(nn.Model):
    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_action(self, observation):
        raise NotImplementedError()

    def reset(self):
        pass

    @property
    def obsfeat_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def recurrent(self):
        """Indicate whether the policy is recurrent"""
        return False


class StochasticPolicy(Policy):
    def __init__(self, env_spec, num_actiondist_params, varscope_name):
        super(StochasticPolicy, self).__init__(env_spec)

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            # Action distribution for current policy
            self._obsfeat_B_Df = tf.placeholder(tf.float32, [batch_size, self.obsfeat_space.shape], name='obsfeat_B_Df') # Df = feature dimensions
            self._actiondist_B_Pa = self._make_actiondist_ops(self._obsfeat_B_Df) # Pa = action distribution params
            self._input_action_B_Da = tf.placeholder(tf.float32, [batch_size, self.action_space.shape], name='input_actions_B_Da') # Action dims
            self._logprobs_B = self._make_actiondist_logprob_ops(self._actiondist_B_Pa, self._input_action_B_Da)

            # proposal distribution from old policy
            self._proposal_actiondist_B_Pa = tf.placeholder(tf.float32, [batch_size, num_actiondist_params], name='proposal_actiondist_B_Pa')
            self._proposal_logprobs_B = self._make_actiondist_logprobs_ops(self._proposal_actiondist_B_Pa, self._input_action_B_Da)

            # Advantage
            self._advantage_B = tf.placeholder(tf.float32, [batch_size], name='advantage_B')

            # Plain pg objective (REINFORCE)
            impweight_B = tf.exp(self._logprobs_B - self._proposal_logprobs_B)
            self._reinfobj = tf.reduce_mean(impweight_B*self._advantage_B)

            # KL
            self._kl_coeff = tf.placeholder(tf.float32, name='kl_cost_coeff')
            kl_B = self._make_actiondist_kl_ops(self._proposal_actiondist_B_Pa, self._actiondist_B_Pa)
            self._kl = tf.reduce_mean(kl_B, 0) # Minimize kl divergence

            # KL Penalty objective for PPO
            self._penobj = self._reinfobj - self._kl_coeff*self._kl

            # All trainable vars done (only _make_* methods)

            # Reading params
            self._param_vars = self.get_trainable_variables()
            self._num_params = self.get_num_params()
            self._curr_params_P = tfutil.flatcat(self._param_vars) # Flatten the params and concat

            # Gradients of objective
            self._reinfobj_grad_P = tfutil.flatcat(tf.gradients(self._reinfobj, self._param_vars))
            self._penobj_grad_P = tfutil.flatcat(tf.gradients(self._penobj, self._param_vars))

            # KL gradient for TRPO
            self._kl_grad_P = tfutil.flatcat(tf.gradients(self._kl, self._param_vars))

            # Check
            self._check_numerics = tf.add_check_numeric_ops()

            # Writing params
            self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
            # For updating vars directly, e.g. for PPO
            self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self._param_vars)

            # Treats placeholder self._flatparams_p as gradient for descent
            with tf.variable_scope('optimizer'):
                self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                vargrads = tfutil.unflatten_into_tensors(self._flatparams_P, [v.get_shape().as_list() for v in self._param_vars])
                self._take_descent_step = tf.train.AdamOptimizer(learning_rate=self._learning_rate).apply_gradients(safezip(vargrads, self._param_vars))

    @property
    def distribution(self):
        raise NotImplementedError()

    def _make_actiondist_ops(self, obsfeat_B_Df):
        """Ops to compute action distribution parameters

        For Gaussian, these would be mean and std
        For categorical, these would be log probabilities
        """
        raise NotImplementedError()

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        raise NotImplementedError()

    def _make_actiondist_kl_ops(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        raise NotImplementedError()

    def compute_action_dist_params(self, sess, obsfeat_B_Df):
        """Actually evaluate action distribution params"""
        return sess.run(self._actiondist_B_Pa, {self._obsfeat_B_Df: obsfeat_B_Df})

    def sample_actions(self, sess, obsfeat_B_Df):
        """Sample actions conditioned on observations
        (Also returns the params)
        """
        return self._sample_from_actiondist(actiondist_B_Pa), actiondist_B_Pa

    def compute_action_logprobs(self, sess, obsfeat_B_Df, actions_B_Da):
        return sess.run(self._logprobs_B, {self._obsfeat_B_Df: obsfeat_B_Df,
                                           self._input_action_B_Da: actions_B_Da})

    def compute_kl_cost(self, sess, proposal_actiondist_B_Pa, obsfeat_B_Df):
        return sess.run(self._kl, {self._obsfeat_B_Df: obsfeat_B_Df,
                                   self._proposal_actiondist_B_Pa: proposal_actiondist_B_Pa})

    Feed = namedtuple('Feed', 'obsfeat_B_Df, actions_B_Da, proposal_actiondist_B_Pa, advantage_B, kl_cost_coeff')
    Feed.__new__.__defaults__ = (None,) * len(Feed._fields) # all default to None
    @staticmethod
    def subsample_feed(feed, size):
        assert feed.obsfeat_B_Df.shape[0] == feed.actions_B_Da.shape[0] == feed.proposal_actiondist_B_Pa.shape[0] == feed.advantage_B.shape[0]
        subsamp_inds = np.random.choice(feed.obsfeat_B_Df.shape[0], size=size)
        return Policy.Feed(
            feed.obsfeat_B_Df[subsamp_inds,:],
            feed.actions_B_Da[subsamp_inds,:],
            feed.proposal_actiondist_B_Pa[subsamp_inds,:],
            feed.advantage_B[subsamp_inds],
            feed.kl_cost_coeff)

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P

    @contextmanager
    def try_params(self, sess, params_D):
        orig_params_D = self.get_params(sess)
        self.set_params(sess, params_D)
        yield                   # Do what you gotta do
        self.set_params(sess, orig_params_D)

    def take_desent_step(self, sess, grad_P, learning_rate):
        sess.run(self._take_descent_step, {self._flatparams_P: grad_P, self._learning_rate: learning_rate})

    ObjInfo = namedtuple('ObjInfo', 'reinfobj, reinfobjgrad_P, kl, klgrad_P, penobj, penobjgrad_P')
    def compute(self, sess, feed, reinfobj=False, reinfobjgrad=False, kl=False, klgrad=False, penobj=False, penobjgrad=False):
        ops = []
        if reinfobj: ops.append(self._reinfobj)
        if reinfobjgrad: ops.append(self._reinfobj_grad_P)
        if kl: ops.append(self._kl)
        if klgrad: ops.append(self._kl_grad_P)
        if penobj: ops.append(self._penobj)
        if penobjgrad: ops.append(self._penobj_grad_P)

        results = sess.run(ops, self._make_feed_dict(self, feed))
        vals = []
        vals.append(results.pop(0) if obj else None)
        vals.append(results.pop(0) if objgrad else None)
        vals.append(results.pop(0) if kl else None)
        vals.append(results.pop(0) if klgrad else None)
        vals.append(results.pop(0) if penobj else None)
        vals.append(results.pop(0) if penobjgrad else None)
        return self.ObjInfo(*vals)
