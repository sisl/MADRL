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


def convert_gym_space(space, n_agents=1):
    if isinstance(space, gym.spaces.Box) or isinstance(space, Box):
        if len(space.shape) > 1:
            assert n_agents == 1, "multi-dimensional inputs for centralized agents not supported"
            return Box(low=np.min(space.low), high=np.max(space.high), shape=space.shape)
        else:
            return Box(low=np.min(space.low), high=np.max(space.high),
                       shape=(space.shape[0] * n_agents,))
    elif isinstance(space, gym.spaces.Discrete) or isinstance(space, Discrete):
        return Discrete(n=space.n**n_agents)
    else:
        raise NotImplementedError


class RLLabEnv(Env, Serializable):

    def __init__(self, env, ma_mode):
        Serializable.quick_init(self, locals())

        self.env = env
        if hasattr(env, 'id'):
            self.env_id = env.id
        else:
            self.env_id = 'MA-Wrapper-v0'

        if ma_mode == 'centralized':
            obsfeat_space = convert_gym_space(env.agents[0].observation_space,
                                              n_agents=len(env.agents))
            action_space = convert_gym_space(env.agents[0].action_space, n_agents=len(env.agents))
        elif ma_mode in ['decentralized', 'concurrent']:
            obsfeat_space = convert_gym_space(env.agents[0].observation_space, n_agents=1)
            action_space = convert_gym_space(env.agents[0].action_space, n_agents=1)

        else:
            raise NotImplementedError

        self._observation_space = obsfeat_space
        self._action_space = action_space
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

    def set_param_values(self, *args, **kwargs):
        self.env.set_param_values(*args, **kwargs)

    def get_param_values(self, *args, **kwargs):
        self.env.get_param_values(*args, **kwargs)
