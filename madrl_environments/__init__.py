from gym import spaces
import numpy as np


class ObservationBuffer(object):

    def __init__(self, env, buffer_size):
        self._env = env
        self._buffer_size = buffer_size
        assert len(env.observation_space.shape) == 1
        bufshape = tuple(env.observation_space.shape) + (buffer_size,)

        self._observation_space = spaces.Box(env.observation_space.low[0],
                                             env.observation_space.high[0], tuple(bufshape))  # XXX
        if env.centralized:
            self._buffer = np.zeros(bufshape)
        else:
            self._buffer = [np.zeros(bufshape) for a in range(env.total_agents)]

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def total_agents(self):
        return self._env.total_agents

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        if self._env.centralized:
            self._buffer[..., 0:self._buffer_size - 1] = self._buffer[..., 1:
                                                                      self._buffer_size].copy()
            self._buffer[..., -1] = obs
            bufobs = self._buffer.copy()
            assert bufobs.shape == self.observation_space.shape, '{} != {}'.format(
                bufobs.shape, self.observation_space.shape)
        else:
            for ag, ag_obs in enumerate(obs):
                self._buffer[ag][..., 0:self._buffer_size - 1] = self._buffer[ag][
                    ..., 1:self._buffer_size].copy()
                self._buffer[ag][..., -1] = ag_obs

            bufobs = [buf.copy() for buf in self._buffer]
        return bufobs, rew, done, info

    def reset(self):
        obs = self._env.reset()

        if self._env.centralized:
            for i in range(self._buffer_size):
                self._buffer[..., i] = obs

            bufobs = self._buffer.copy()
            assert bufobs.shape == self.observation_space.shape, '{} != {}'.format(
                bufobs.shape, self.observation_space.shape)
        else:
            assert isinstance(obs, list)
            for ag, ag_obs in enumerate(obs):
                for i in range(self._buffer_size):
                    self._buffer[ag][..., i] = ag_obs

            bufobs = [buf.copy() for buf in self._buffer]

        return bufobs

    def render(self, *args, **kwargs):
        return self._env.render(args, kwargs)
