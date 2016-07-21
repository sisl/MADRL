import glob
import os
from os.path import join
from subprocess import call

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib.patches import Rectangle

from .utils import agent_utils
from .utils.AgentLayer import AgentLayer
from .utils.Controllers import RandomPolicy

#################################################################
# Implements an Evade Pursuit Problem in 2D
#################################################################


class DecPursuitEvade():

    def __init__(self, map_matrix, **kwargs):
        """
        In evade purusit a set of pursuers must 'tag' a set of evaders
        Required arguments:
        - map_matrix: the map on which agents interact

        Optional arguments:
        - Ally layer: list of pursuers
        Opponent layer: list of evaders
        Ally controller: stationary policy of ally pursuers
        Ally controller: stationary policy of opponent evaders
        map_matrix: the map on which agents interact
        catchr: reward for 'tagging' a single evader
        caughtr: reward for getting 'tagged' by a pursuer
        train_pursuit: flag indicating if we are simulating pursuers or evaders
        initial_config: dictionary of form
            initial_config['allies']: the initial ally confidguration (matrix)
            initial_config['opponents']: the initial opponent confidguration (matrix)
        """

        xs, ys = map_matrix.shape
        self.map_matrix = map_matrix
        self.xs = xs
        self.ys = ys

        self.n_evaders = kwargs.pop('n_evaders', 1)
        self.n_pursuers = kwargs.pop('n_pursuers', 1)

        self.obs_range = kwargs.pop('obs_range', 3)  # can see 3 grids around them by default
        self.obs_size = self.obs_range**2
        self.obs_offset = int((self.obs_range - 1) / 2)

        pursuers = agent_utils.create_agents(self.n_pursuers, map_matrix)
        evaders = agent_utils.create_agents(self.n_evaders, map_matrix)

        self.pursuer_layer = kwargs.pop('ally_layer', AgentLayer(xs, ys, pursuers))
        self.evader_layer = kwargs.pop('opponent_layer', AgentLayer(xs, ys, evaders))

        self.layer_norm = kwargs.pop('layer_norm', 10)
        self.flatten = kwargs.pop('flatten', True)

        self.n_catch = kwargs.pop('n_catch', 2)

        self.plt_delay = kwargs.pop('plt_delay', 1.0)

        self.random_opponents = kwargs.pop('random_opponents', False)
        self.max_opponents = kwargs.pop('max_opponents', 10)

        n_act_purs = self.pursuer_layer.get_nactions(0)
        n_act_ev = self.evader_layer.get_nactions(0)

        self.evader_controller = kwargs.pop('evader_controller', RandomPolicy(n_act_purs))
        self.pursuer_controller = kwargs.pop('pursuer_controller', RandomPolicy(n_act_ev))

        self.current_agent_layer = np.zeros((xs, ys), dtype=np.int32)

        self.catchr = kwargs.pop('catchr', 0.01)
        self.caughtr = kwargs.pop('caughtr', -0.01)

        self.term_pursuit = kwargs.pop('term_pursuit', 1.0)
        self.term_evade = kwargs.pop('term_evade', -1.0)

        self.urgency_reward = kwargs.pop('urgency_reward', 0.0)

        self.ally_actions = np.zeros(n_act_purs, dtype=np.int32)
        self.opponent_actions = np.zeros(n_act_ev, dtype=np.int32)

        self.train_pursuit = kwargs.pop('train_pursuit', True)

        if self.train_pursuit:
            self.low = np.array([0.0 for i in xrange(3 * self.obs_range**2)])
            self.high = np.array([1.0 for i in xrange(3 * self.obs_range**2)])
            self.action_space = spaces.Discrete(n_act_purs)
            self.observation_space = spaces.Box(self.low, self.high)
            self.local_obs = np.zeros(
                (self.n_pursuers, 3, self.obs_range, self.obs_range))  # Nagents X 3 X xsize X ysize
            self.act_dims = [n_act_purs for i in xrange(self.n_pursuers)]
            self.total_agents = self.n_pursuers
        else:
            self.low = np.array([0.0 for i in xrange(3 * self.obs_range**2)])
            self.high = np.array([1.0 for i in xrange(3 * self.obs_range**2)])
            self.action_space = spaces.Discrete(n_act_ev)
            self.observation_space = spaces.Box(self.low, self.high)
            self.local_obs = np.zeros(
                (self.n_evaders, 3, self.obs_range, self.obs_range))  # Nagents X 3 X xsize X ysize
            self.act_dims = [n_act_purs for i in xrange(self.n_evaders)]
            self.total_agents = self.n_evaders
        self.pursuers_gone = np.array([False for i in xrange(self.n_pursuers)])
        self.evaders_gone = np.array([False for i in xrange(self.n_evaders)])

        self.initial_config = kwargs.pop('initial_config', {})

        self.current_agent = kwargs.pop('current_agent', 0)

        self.model_dims = (4,) + map_matrix.shape

        self.n_actions = n_act_purs

        self.model_state = np.zeros(self.model_dims, dtype=np.float32)

    #################################################################
    # The functions below are the interface with MultiAgentSiulator # 
    #################################################################

    def reset(self):
        self.pursuers_gone.fill(False)
        self.evaders_gone.fill(False)
        if self.random_opponents:
            if self.train_pursuit:
                self.n_evaders = np.random.randint(1, self.max_opponents)
            else:
                self.n_pursuers = np.random.randint(1, self.max_opponents)

        self.pursuer_layer = AgentLayer(
            self.xs, self.ys,
            agent_utils.create_agents(self.n_pursuers, self.map_matrix, randinit=True))
        self.evader_layer = AgentLayer(
            self.xs, self.ys,
            agent_utils.create_agents(self.n_evaders, self.map_matrix, randinit=True))
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()
        if self.train_pursuit:
            return self.collect_obs(self.pursuer_layer, self.pursuers_gone)
        else:
            return self.collect_obs(self.evader_layer, self.evaders_gone)

    def step(self, actions):
        """
            Step the system forward. Actions is an iterable of action indecies.
        """
        r = self.reward()

        if self.train_pursuit:
            agent_layer = self.pursuer_layer
            opponent_layer = self.evader_layer
            opponent_controller = self.evader_controller
            gone_flags = self.pursuers_gone
        else:
            agent_layer = self.evader_layer
            opponent_layer = self.pursuer_layer
            opponent_controller = self.pursuer_controller
            gone_flags = self.evaders_gone

        # move allies
        for i, a in enumerate(actions):
            agent_layer.move_agent(i, a)

        # move opponents
        for i in xrange(opponent_layer.n_agents()):
            # controller input should be an observation, but doesn't matter right now
            action = opponent_controller.act(self.model_state)
            opponent_layer.move_agent(i, action)

        # model state always has form: map, purusers, opponents, current agent id
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()

        # remove agents that are caught
        ev_remove, pr_remove = self.remove_agents()

        o = self.collect_obs(agent_layer, gone_flags)

        # add caught rewards
        if self.train_pursuit:
            r += (ev_remove * self.term_pursuit)
        else:
            r += (pr_remove * self.term_evade)
        r += self.urgency_reward

        done = self.is_terminal()

        return o, r, done, None

    def render(self):
        plt.matshow(self.model_state[0].T, cmap=plt.get_cmap('Greys'), fignum=1)
        for i in xrange(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            plt.plot(x, y, "r*", markersize=12)
            if self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range,
                                       alpha=0.5, facecolor="#FF9848"))
        for i in xrange(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            plt.plot(x, y, "b*", markersize=12)
            if not self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range,
                                       alpha=0.5, facecolor="#009ACD"))
        plt.pause(self.plt_delay)
        plt.clf()

    def animate(self, policy, nsteps, file_name, rate=1.5):
        """
            Save an animation to an mp4 file.
        """
        plt.figure(0)
        # run sim loop
        o = self.reset()
        file_path = "/".join(file_name.split("/")[0:-1])
        temp_name = join(file_path, "temp_0.png")
        # generate .pngs
        self.save_image(temp_name)
        for i in xrange(nsteps):
            a, adist = policy.act(o)
            o, r, done, _ = self.step(a)
            temp_name = join(file_path, "temp_" + str(i + 1) + ".png")
            self.save_image(temp_name)
            if done:
                break
        # use ffmpeg to create .pngs to .mp4 movie
        ffmpeg_cmd = "ffmpeg -framerate " + str(rate) + " -i " + join(
            file_path, "temp_%d.png") + " -c:v libx264 -pix_fmt yuv420p " + file_name
        call(ffmpeg_cmd.split())
        # clean-up by removing .pngs
        map(os.remove, glob.glob(join(file_path, "temp_*")))

    def save_image(self, file_name):
        plt.cla()
        plt.matshow(self.model_state[0].T, cmap=plt.get_cmap('Greys'), fignum=0)
        x, y = self.pursuer_layer.get_position(0)
        plt.plot(x, y, "r*", markersize=12)
        for i in xrange(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            plt.plot(x, y, "r*", markersize=12)
            if self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range,
                                       alpha=0.5, facecolor="#FF9848"))
        for i in xrange(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            plt.plot(x, y, "b*", markersize=12)
            if not self.train_pursuit:
                ax = plt.gca()
                ofst = self.obs_range / 2.0
                ax.add_patch(Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range,
                                       alpha=0.5, facecolor="#009ACD"))

        xl, xh = -self.obs_offset - 1, self.xs + self.obs_offset + 1
        yl, yh = -self.obs_offset - 1, self.ys + self.obs_offset + 1
        plt.xlim([xl, xh])
        plt.ylim([yl, yh])
        plt.savefig(file_name)

    def sample_action(self):
        # returns a list of actions
        actions = []
        for i in xrange(self.pursuer_layer.n_agents()):
            actions.append(self.action_space.sample())
        return actions

    def reward(self):
        r = self.pursuer_reward() if self.train_pursuit else self.evader_reward()
        return r

    def is_terminal(self):
        #ev = self.evader_layer.get_state_matrix()  # evader positions
        #if np.sum(ev) == 0.0:
        if self.evader_layer.n_agents() == 0:
            return True
        return False

    def update_ally_controller(self, controller):
        self.ally_controller = controller

    def update_opponent_controller(self, controller):
        self.opponent_controller = controller

    def set_agents(self, agent_type):
        if agent_type == "allies":
            self.train_pursuit = True
        else:
            self.train_pursuit = False

    #################################################################

    def n_agents(self):
        n = self.pursuer_layer.n_agents() if self.train_pursuit else self.evader_layer.n_agents()
        return n

    def collect_obs(self, agent_layer, gone_flags):
        obs = []
        nage = 0
        for i in xrange(self.total_agents):
            if gone_flags[i]: 
                obs.append(None)
            else:
                o = self.collect_obs_by_idx(agent_layer, nage)
                obs.append(o)
                nage += 1
        return obs

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        n = agent_layer.n_agents()
        self.local_obs.fill(-0.1)  # border walls set to -0.1?
        # loop through agents
        # get the obs bounds
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        self.local_obs[agent_idx, :, xolo:xohi, yolo:yohi] = self.model_state[0:3, xlo:xhi, ylo:yhi]
        if self.flatten:
            return self.local_obs[agent_idx].flatten() / self.layer_norm
        return self.local_obs[agent_idx] / self.layer_norm

    def obs_clip(self, x, y):
        # :( this is a mess, beter way to do the slicing?
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (np.clip(xld, 0, self.xs - 1), np.clip(xhd, 0, self.xs - 1),
                              np.clip(yld, 0, self.ys - 1), np.clip(yhd, 0, self.ys - 1))
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(np.clip(yld, -self.obs_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def pursuer_reward(self):
        """
        Computes the joint reward for pursuers
        """
        # rewarded for each tagged evader
        ps = self.pursuer_layer.get_state_matrix()  # pursuer positions
        es = self.evader_layer.get_state_matrix()  # evader positions
        tagged = np.sum((ps > 0) * es)  # number of tagged evaders
        rtot = self.catchr * tagged
        return rtot

    def evader_reward(self):
        """
        Computes the joint reward for evaders
        """
        # penalized for each tagged evader
        ps = self.pursuer_layer.get_state_matrix()  # pursuer positions
        es = self.evader_layer.get_state_matrix()  # evader positions
        tagged = np.sum((ps > 0) * es)  # number of tagged evaders
        rtot = self.caughtr * tagged
        return rtot

    def remove_agents(self):
        """
        Remove agents that are caught. Return tuple (n_evader_removed, n_pursuer_removed)
        """
        n_pursuer_removed = 0
        n_evader_removed = 0
        removed_evade = []
        removed_pursuit = []

        ai = 0
        rems = 0
        for i in xrange(self.n_evaders):
            if self.evaders_gone[i]: continue
            x, y = self.evader_layer.get_position(ai)
            if self.model_state[1, x, y] >= self.n_catch:
                # add prob remove?
                removed_evade.append(ai - rems)
                self.evaders_gone[i] = True
                rems += 1
            ai += 1


        ai = 0
        for i in xrange(self.pursuer_layer.n_agents()):
            if self.pursuers_gone[i]: continue
            x, y = self.pursuer_layer.get_position(ai)
            # number of evaders > 0 and number of pursuers < n_catch
            #if self.model_state[2,x,y] > 0 and self.model_state[1,x,y] < self.n_catch:
            # probabilistic model for this
            # add prob remove?
            # removed_pursuit.append(i-rems)
            # rems += 1
            #print "Removing evader:", x, y, i-rems
        for ridx in removed_evade:
            self.evader_layer.remove_agent(ridx)
            n_evader_removed += 1
        for ridx in removed_pursuit:
            self.pursuer_layer.remove_agent(ridx)
            n_pursuer_removed += 1
        return n_evader_removed, n_pursuer_removed

    def get_layers_pursuer(self, agent_idx):
        """
        Return a 4-tuple of the form: (building layer, opponent layer, ally layer, agent of interest layer)
        Each layer is a 2D numpy array of same size as map_matrix
        """
        agent_state = self.current_agent_layer
        agent_state.fill(0)
        (x, y) = self.ally_layer.get_position(agent_idx)
        agent_state[x, y] = 1
        self.model_state[0] = self.map_matrix
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state_matrix(
        ), self.ally_layer.get_state_matrix()
        self.model_state[3] = agent_state
        return self.model_state

    def get_layers_evader(self, agent_idx):
        """
        Return a 4-tuple of the form: (building layer, opponent layer, ally layer, agent of interest layer)
        Each layer is a 2D numpy array of same size as map_matrix
        """
        agent_state = self.current_agent_layer
        agent_state.fill(0)
        (x, y) = self.opponent_layer.get_position(agent_idx)
        agent_state[x, y] = 1
        self.model_state[0] = self.map_matrix
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state_matrix(
        ), self.ally_layer.get_state_matrix()
        self.model_state[3] = agent_state
        return self.model_state

    def transition_pursuer(self, agent_idx, action):
        pursuers = self.ally_layer
        evaders = self.opponent_layer

        pursuer_actions = self.ally_actions
        evader_actions = self.opponent_actions

        # get the pursuer actions
        for i in xrange(self.n_pursuers):
            if i == agent_idx:
                continue
            state = self.get_layers_pursuer(i)
            pursuer_actions[i] = self.ally_controller.action(state)

        # get evader actions
        for i in xrange(self.n_evaders):
            state = self.get_layers_evader(i)
            evader_actions[i] = self.opponent_controller.action(state)

        # evolve the system
        for i in xrange(self.n_pursuers):
            if i == agent_idx:
                continue
            pursuers.move_agent(i, pursuer_actions[i])
        for i in xrange(self.n_evaders):
            evaders.move_agent(i, evader_actions[i])

        # move the agent in question
        pursuers.move_agent(agent_idx, action)

    def transition_evader(self, agent_idx, action):
        pursuers = self.ally_layer
        evaders = self.opponent_layer

        pursuer_actions = self.ally_actions
        evader_actions = self.opponent_actions

        # get the pursuer actions
        for i in xrange(self.n_pursuers):
            state = self.get_layers_pursuer(i)
            pursuer_actions[i] = self.ally_controller.action(state)

        # get evader actions
        for i in xrange(self.n_evaders):
            if i == agent_idx:
                continue
            state = self.get_layers_evader(i)
            evader_actions[i] = self.opponent_controller.action(state)

        # evolve the system
        for i in xrange(self.n_pursuers):
            pursuers.move_agent(i, pursuer_actions[i])
        for i in xrange(self.n_evaders):
            if i == agent_idx:
                continue
            evaders.move_agent(i, evader_actions[i])

        # move the agent in question
        evaders.move_agent(agent_idx, action)

    def transition_all(self):
        pursuers = self.ally_layer
        evaders = self.opponent_layer

        pursuer_actions = self.ally_actions
        evader_actions = self.opponent_actions

        # get the pursuer actions
        for i in xrange(self.n_pursuers):
            state = self.get_layers_pursuer(i)
            pursuer_actions[i] = self.ally_controller.action(state)

        # get evader actions
        for i in xrange(self.n_evaders):
            state = self.get_layers_evader(i)
            evader_actions[i] = self.opponent_controller.action(state)

        # evolve the system
        for i in xrange(self.n_pursuers):
            pursuers.move_agent(i, pursuer_actions[i])
        for i in xrange(self.n_evaders):
            evaders.move_agent(i, evader_actions[i])
