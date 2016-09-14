import numpy as np
import math as m

from rltools.policy import Policy

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4

class PursuitHeuristicPolicy(Policy):

    def __init__(self, observation_space, action_space):
        super(PursuitHeuristicPolicy, self).__init__(observation_space, action_space)

    def sample_actions(self, obs_B_Do, deterministic=True):

        n_ev = np.sum(obs_B_Do[:,:,2])
        n_pr = np.sum(obs_B_Do[:,:,1])
        xs, ys = obs_B_Do.shape[0], obs_B_Do.shape[1]

        x, y = xs/2, ys/2

        # if see evader move to it
        if n_ev > 0:
            xev, yev = np.nonzero(obs_B_Do[:,:,2])
            d = np.sqrt((xev-x)**2 + (yev-y)**2)
            midx = np.argmin(d)
            xc, yc = xev[midx], yev[midx]
            if xc == x and yc == y:
                return STAY, None
            ang = m.atan2(yc-y, xc-x)
            ang = (ang + np.pi) % (2*np.pi) - np.pi
            # Right
            if -np.pi/4 <= ang < np.pi/4:
                return RIGHT, None
            # Up
            elif np.pi/4 <= ang < 3/4.*np.pi:
                return UP, None
            # Left
            elif ang >= 3/4.*np.pi or ang < -3/4.*np.pi:
                return LEFT, None
            # Down
            elif -3/4.*np.pi <= ang < -np.pi/4:
                return DOWN, None
            else:
                return self.action_space.sample(), None

        return self.action_space.sample(), None



if __name__ == '__main__':
    from madrl_environments.pursuit import PursuitEvade
    from madrl_environments.pursuit.utils import *
    from vis import Visualizer
    
    map_mat = TwoDMaps.rectangle_map(10, 10)

    env = PursuitEvade([map_mat], n_evaders=6, n_pursuers=5, obs_range=7, n_catch=2, surround=False, flatten=False)

    policy = PursuitHeuristicPolicy(env.agents[0].observation_space, env.agents[0].action_space)

    obs = env.reset()
    rew = 0.0
    for _ in xrange(500):
        env.render()
        act_list = []
        for o in obs:
            a, _ = policy.sample_actions(o)
            act_list.append(a)
        obs, r, done, _ = env.step(act_list)
        rew += np.mean(r)
        if done: break


    pprint.pprint(rew)
    pprint.pprint(info)
