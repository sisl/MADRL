from gym import spaces
import numpy as np


class Agent(object):

    def __new__(cls, *args, **kwargs):
        agent = super(Agent, cls).__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class AbstractMAEnv(object):

    def __new__(cls, *args, **kwargs):
        # Easier to override __init__
        env = super(AbstractMAEnv, cls).__new__(cls)
        env._unwrapped = None
        return env

    def seed(self, seed=None):
        return []

    @property
    def agents(self):
        """Returns the agents in the environment. List of objects inherited from Agent class

        Should give us information about cooperating and competing agents?
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the game"""
        raise NotImplementedError()

    def step(self, actions):
        raise NotImplementedError()

    @property
    def is_terminal(self):
        raise NotImplementedError()

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def animate(self, act_fn, nsteps, **kwargs):
        """act_fn could be a list of functions for each agent in the environemnt that we can control"""
        if not isinstance(act_fn, list):
            act_fn = [act_fn for _ in range(len(self.agents))]
        assert len(act_fn) == len(self.agents)
        obs = self.reset()
        self.render(**kwargs)
        rew = np.zeros((len(self.agents)))
        for step in range(nsteps):
            a = map(lambda afn, o: afn(o), act_fn, obs)
            obs, r, done, _ = self.step(a)
            rew += r
            self.render(**kwargs)
            if done:
                break
        return rew

    @property
    def unwrapped(self):
        if self._unwrapped is not None:
            return self._unwrapped
        else:
            return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class WrappedAgent(Agent):

    def __init__(self, agent, new_observation_space):
        self._unwrapped = agent
        self._new_observation_space = new_observation_space

    @property
    def observation_space(self):
        return self._new_observation_space

    @property
    def action_space(self):
        return self._unwrapped.action_space

    def unwrapped(self):
        if self.unwrapped is not None:
            return self._unwrapped
        else:
            return self


class ObservationBuffer(AbstractMAEnv):

    def __init__(self, env, buffer_size):
        self._unwrapped = env
        self._buffer_size = buffer_size
        assert all([len(agent.observation_space.shape) == 1 for agent in env.agents])  # XXX
        bufshapes = [tuple(agent.observation_space.shape) + (buffer_size,) for agent in env.agents]
        self._buffer = [np.zeros(bufshape) for bufshape in bufshapes]
        self.reward_mech = self._unwrapped.reward_mech

    @property
    def agents(self):
        aglist = []
        for agid, agent in enumerate(self._unwrapped.agents):
            if isinstance(agent.observation_space, spaces.Box):
                newobservation_space = spaces.Box(low=agent.observation_space.low[0],
                                                  high=agent.observation_space.high[0],
                                                  shape=self._buffer[agid].shape)
            # elif isinstance(agent.observation_sapce, spaces.Discrete):
            else:
                raise NotImplementedError()

            aglist.append(WrappedAgent(agent, newobservation_space))

        return aglist

    def seed(self, seed=None):
        return self._unwrapped.seed(seed)

    def step(self, action):
        obs, rew, done, info = self._unwrapped.step(action)
        for agid, agid_obs in enumerate(obs):
            self._buffer[agid][..., 0:self._buffer_size - 1] = self._buffer[agid][
                ..., 1:self._buffer_size].copy()
            self._buffer[agid][..., -1] = agid_obs

        bufobs = [buf.copy() for buf in self._buffer]
        return bufobs, rew, done, info

    def reset(self):
        obs = self._unwrapped.reset()

        assert isinstance(obs, list)
        for agid, agid_obs in enumerate(obs):
            for i in range(self._buffer_size):
                self._buffer[agid][..., i] = agid_obs

        bufobs = [buf.copy() for buf in self._buffer]
        return bufobs

    def render(self, *args, **kwargs):
        return self._unwrapped.render(*args, **kwargs)

    def animate(self, *args, **kwargs):
        return self._unwrapped.animate(*args, **kwargs)
