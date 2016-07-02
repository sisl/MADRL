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
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def recurrent(self):
        """Indicate whether the policy is recurrent"""
        return False


class StochasticPolicy(Policy):
    def __init__(self, env_spec):
        Policy.__init__(env_spec)


    @property
    def distribution(self):
        raise NotImplementedError()

    def _make_actiondist_ops(self, obsfeat_B_Df):
        """Ops to compute action distribution parameters

        For Gaussian, these would be mean and std
        For categorical, these would be log probabilities
        """
        raise NotImplementedError()

    def _make_actiondist_logprobs_os(self, actiondist_B_Pa, input_actions_B_Da):
        raise NotImplementedError()

    def _make_actiondist_kl_ops(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        raise NotImplementedError()

    def compute_action_dist_params(self, sess, obsfeat_B_Df):
        """Actually evaluate action distribution params"""
        raise NotImplementedError()

    def sample_actions(self, sess, obsfeat_B_Df):
        raise NotImplementedError()

    def compute_action_logprobs(self, sess, obsfeat_B_Df, actions_B_Da):
        raise NotImplementedError()

    def compute_kl_cost(self, sess, proposal_actiondist_B_Pa, obsfeat_B_Df):
        raise NotImplementedError()
