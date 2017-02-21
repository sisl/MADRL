from rltools.util import EzPickle, stack_dict_list
from gym import spaces, error
from gym.monitoring.video_recorder import ImageEncoder
import numpy as np
import time
import scipy


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

    @property
    def reward_mech(self):
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
        encoder = None
        vid_loc = kwargs.pop('vid', None)
        obs = self.reset()
        frame = self.render(**kwargs)
        if vid_loc:
            fps = kwargs.pop('fps', 30)
            encoder = ImageEncoder(vid_loc, frame.shape, fps)
            try:
                encoder.capture_frame(frame)
            except error.InvalidFrame as e:
                print('Invalid video frame, {}'.format(e))

        rew = np.zeros((len(self.agents)))
        traj_info_list = []
        for step in range(nsteps):
            a = map(lambda afn, o: afn(o), act_fn, obs)
            obs, r, done, info = self.step(a)
            rew += r
            if info:
                traj_info_list.append(info)

            frame = self.render(**kwargs)
            if vid_loc:
                try:
                    encoder.capture_frame(frame)
                except error.InvalidFrame as e:
                    print('Invalid video frame, {}'.format(e))

            if done:
                break

        traj_info = stack_dict_list(traj_info_list)
        return rew, traj_info

    def update_curriculum(self, itr):
        """Updates curriculum learning parameters"""
        return

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

    @property
    def reward_mech(self):
        return self._unwrapped.reward_mech

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


class StandardizedEnv(AbstractMAEnv, EzPickle):

    def __init__(self, env, scale_reward=1., enable_obsnorm=False, enable_rewnorm=False,
                 obs_alpha=0.001, rew_alpha=0.001, eps=1e-8):
        EzPickle.__init__(self, env, scale_reward, enable_obsnorm, enable_rewnorm, obs_alpha,
                          rew_alpha, eps)
        self._unwrapped = env
        self._scale_reward = scale_reward
        self._enable_obsnorm = enable_obsnorm
        self._enable_rewnorm = enable_rewnorm
        self._obs_alpha = obs_alpha
        self._rew_alpha = rew_alpha
        self._eps = eps
        self._flatobs_shape = [None for _ in env.agents]
        self._obs_mean = [None for _ in env.agents]
        self._obs_var = [None for _ in env.agents]
        self._rew_mean = [None for _ in env.agents]
        self._rew_var = [None for _ in env.agents]

        for agid, agent in enumerate(env.agents):
            if isinstance(agent.observation_space, spaces.Box):
                self._flatobs_shape[agid] = np.prod(agent.observation_space.shape)
            elif isinstance(env.observation_space, spaces.Discrete):
                self._flatobs_shape[agid] = agent.observation_space.n

            self._obs_mean[agid] = np.zeros(self._flatobs_shape[agid])
            self._obs_var[agid] = np.ones(self._flatobs_shape[agid])
            self._rew_mean[agid] = 0.
            self._rew_var[agid] = 1.

    @property
    def reward_mech(self):
        return self._unwrapped.reward_mech

    @property
    def agents(self):
        return self._unwrapped.agents

    def update_obs_estimate(self, observations):
        for agid, obs in enumerate(observations):
            flatobs = np.asarray(obs).flatten()
            self._obs_mean[agid] = (1 - self._obs_alpha
                                   ) * self._obs_mean[agid] + self._obs_alpha * flatobs
            self._obs_var[agid] = (
                1 - self._obs_alpha
            ) * self._obs_var[agid] + self._obs_alpha * np.square(flatobs - self._obs_mean[agid])

    def update_rew_estimate(self, rewards):
        for agid, reward in enumerate(rewards):
            self._rew_mean[agid] = (1 - self._rew_alpha
                                   ) * self._rew_mean[agid] + self._rew_alpha * reward
            self._rew_var[agid] = (
                1 - self._rew_alpha
            ) * self._rew_var[agid] + self._rew_alpha * np.square(reward - self._rew_mean[agid])

    def standardize_obs(self, observation):
        assert isinstance(observation, list)
        self.update_obs_estimate(observation)
        return [(obs - obsmean) / (np.sqrt(obsvar) + self._eps)
                for (obs, obsmean, obsvar) in zip(observation, self._obs_mean, self._obs_var)]

    def standardize_rew(self, reward):
        assert isinstance(reward, (list, np.ndarray))
        self.update_rew_estimate(reward)
        return [
            rew / (np.sqrt(rewvar) + self._eps)
            for (rew, rewmean, rewvar) in zip(reward, self._rew_mean, self._rew_var)
        ]

    def seed(self, seed=None):
        return self._unwrapped.seed(seed)

    def reset(self):
        obs = self._unwrapped.reset()
        if self._enable_obsnorm:
            return self.standardize_obs(obs)
        else:
            return obs

    def step(self, *args):
        nobslist, rewardlist, done, info = self._unwrapped.step(*args)
        if self._enable_obsnorm:
            nobslist = self.standardize_obs(nobslist)
        if self._enable_rewnorm:
            rewardlist = self.standardize_rew(rewardlist)

        rewardlist = [self._scale_reward * rew for rew in rewardlist]
        return nobslist, rewardlist, done, info

    def __getstate__(self):
        d = EzPickle.__getstate__(self)
        d['_obs_mean'] = self._obs_mean
        d['_obs_var'] = self._obs_var
        return d

    def __setstate__(self, d):
        EzPickle.__setstate__(self, d)
        self._obs_mean = d['_obs_mean']
        self._obs_var = d['_obs_var']

    def __str__(self):
        return "Normalized {}".format(self._unwrapped)

    def render(self, *args, **kwargs):
        return self._unwrapped.render(*args, **kwargs)

    def animate(self, *args, **kwargs):
        return self._unwrapped.animate(*args, **kwargs)


class DiagnosticsWrapper(AbstractMAEnv, EzPickle):

    def __init__(self, env, discount=0.99, log_interval=501):
        self._unwrapped = env
        self._discount = discount
        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = np.zeros((len(env.agents)))
        self._episode_length = 0
        self._all_rewards = []

    def reset(self):
        obs = self._unwrapped.reset()
        self._episode_length = 0
        self._episode_reward = np.zeros((len(self._unwrapped.agents)))
        self._all_rewards = []
        return obs

    def step(self, *args):
        obslist, rewardlist, done, info = self._unwrapped.step(*args)
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            to_log['diagnostics/fps'] = fps
            self._last_time = cur_time

        if rewardlist is not None:
            self._episode_reward += np.asarray(rewardlist)
            self._episode_length += 1
            self._all_rewards.append(rewardlist)

        if done:
            total_time = time.time() - self._episode_time
            for agid, epr in enumerate(self._episode_reward):
                to_log['global/episode_reward_agent{}'.format(agid)] = epr

            to_log['global/episode_avg_reward'] = np.mean(self._episode_reward)
            arr = np.asarray(self._all_rewards).mean(axis=1)
            to_log['global/episode_disc_return'] = _discount_sum(arr, self._discount)

            to_log['global/episode_length'] = self._episode_length
            to_log['global/episode_time'] = total_time
            self._episode_length = 0
            self._episode_reward = np.zeros_like(self._episode_reward)
            self._all_rewards = []

        return obslist, rewardlist, done, to_log

    def seed(self, seed=None):
        return self._unwrapped.seed(seed)

    def render(self, *args, **kwargs):
        return self._unwrapped.render(*args, **kwargs)

    def animate(self, *args, **kwargs):
        return self._unwrapped.animate(*args, **kwargs)

    @property
    def reward_mech(self):
        return self._unwrapped.reward_mech

    @property
    def agents(self):
        return self._unwrapped.agents


def _discount_sum(x, discount):
    return np.sum(x * (discount**np.arange(len(x))))
