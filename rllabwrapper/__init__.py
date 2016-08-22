from __future__ import print_function
from __future__ import absolute_import

import gym
import gym.envs
import gym.spaces

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

import numpy as np


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class RLLabEnv(Env, Serializable):

    def __init__(self, env):
        Serializable.quick_init(self, locals())

        self.env = env
        if hasattr(env, 'id'):
            self.env_id = env.id
        else:
            self.env_id = 'MA-Wrapper-v0'

        self._observation_space = convert_gym_space(env.agents[0].observation_space)
        self._action_space = convert_gym_space(env.agents[0].action_space)
        if hasattr(env, 'timestep_limit'):
            self._horizon = env.timestep_limit
        else:
            self._horizon = 250

    @property
    def agents(self):
        return self.env.agents

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if info is None:
            info = dict()
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()
