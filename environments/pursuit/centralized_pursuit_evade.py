import numpy as np
from gym import spaces

from utils import agent_utils
from utils.Controllers import RandomPolicy
from utils.AgentLayer import AgentLayer


#################################################################
# Implements an Evade Pursuit Problem in 2D
#################################################################

class CentralizedPursuitEvade():

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

        self.obs_range = kwargs.pop('obs_range', 3) # can see 3 grids around them by default
        self.obs_size = self.obs_range**2
        self.obs_offset = int((self.obs_range - 1) / 2)

        pursuers = agent_utils.create_agents(self.n_pursuers, map_matrix)
        evaders = agent_utils.create_agents(self.n_evaders, map_matrix)

        self.pursuer_layer = kwargs.pop('ally_layer', AgentLayer(xs, ys, pursuers))
        self.evader_layer = kwargs.pop('opponent_layer', AgentLayer(xs, ys, evaders)) 

        self.layer_norm = kwargs.pop('layer_norm', 10)
        self.flatten = kwargs.pop('flatten', True)

        self.n_catch = kwargs.pop('n_catch', 2)

        self.low = np.array([0.0 for i in xrange(3 * self.obs_range**2 * self.n_pursuers)])
        self.high = np.array([1.0 for i in xrange(3 * self.obs_range**2 * self.n_pursuers)])

        n_act_purs = self.pursuer_layer.get_nactions(0)
        n_act_ev = self.evader_layer.get_nactions(0)

        self.action_space = spaces.Discrete(n_act_purs**self.n_pursuers)
        self.observation_space = spaces.Box(self.low, self.high)

        self.act_dims = [n_act_purs for i in xrange(self.n_pursuers)]

        self.local_obs = np.zeros((self.n_pursuers, 3, self.obs_range, self.obs_range)) # Nagents X 3 X xsize X ysize

        self.evader_controller = kwargs.pop('ally_controller', RandomPolicy(n_act_purs))
        self.pursuer_controller = kwargs.pop('opponent_controller', RandomPolicy(n_act_ev)) 



        self.current_agent_layer = np.zeros((xs, ys), dtype=np.int32)

        self.catchr = kwargs.pop('catchr', 0.01)
        self.caughtr = kwargs.pop('caughtr', -0.01)

        self.term_pursuit = kwargs.pop('term_pursuit', 1.0)
        self.term_evade = kwargs.pop('term_evade', -1.0)

        self.ally_actions = np.zeros(n_act_purs, dtype=np.int32)
        self.opponent_actions = np.zeros(n_act_ev, dtype=np.int32)

        self.train_pursuit = kwargs.pop('train_pursuit', True)

        self.initial_config = kwargs.pop('initial_config', {})

        self.model_dims = (4,) + map_matrix.shape

        self.n_actions = n_act_purs

        self.model_state = np.zeros(self.model_dims, dtype=np.float32)

    #################################################################
    # The functions below are the interface with MultiAgentSiulator # 
    #################################################################

    def reset(self):
        self.pursuer_layer = AgentLayer(self.xs, self.ys, 
                                agent_utils.create_agents(self.n_pursuers, self.map_matrix, randinit=True))
        self.evader_layer = AgentLayer(self.xs, self.ys,
                                  agent_utils.create_agents(self.n_evaders, self.map_matrix, randinit=True))
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()
        return self.collect_obs(self.pursuer_layer)



    def step(self, actions):
        """
            Step the system forward. Actions is an iterable of action indecies.
        """

        r = self.reward()

        if actions is list:
            # move all agents
            for i, a in enumerate(actions):
                self.pursuer_layer.move_agent(i, a)
        else:
            # ravel it up
            act_idxs = np.unravel_index(actions, self.act_dims)
            for i, a in enumerate(act_idxs):
                self.pursuer_layer.move_agent(i, a)

        for i in xrange(self.evader_layer.n_agents()):
            # controller input should be an observation, but doesn't matter right now
            action = self.evader_controller.act(self.model_state)
            self.evader_layer.move_agent(i, action)

        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()

        # remove agents that are caught
        ev_remove, pr_remove = self.remove_agents()

        # add caught rewards
        r += (ev_remove * self.term_pursuit)

        o = self.collect_obs(self.pursuer_layer)

        done = self.is_terminal()

        return o, r, done, None


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
        ev = self.evader_layer.get_state_matrix() # evader positions
        if np.sum(ev) == 0.0:
            return True
        return False




    def act(self, agent_idx, action):
        if self.train_pursuit:
            self.transition_pursuer(agent_idx, action)
        else:
            self.transition_evader(agent_idx, action)

    def observe(self, agent_idx):
        if self.train_pursuit:
            return self.get_layers_pursuer(agent_idx)
        else:
            return self.get_layers_evader(agent_idx)

    def update_ally_controller(self, controller):
        self.ally_controller = controller

    def update_opponent_controller(self, controller):
        self.opponent_controller = controller


    def initialize(self, origin = False, random = True):
        mmatrix = self.map_matrix
        n_pursuers = self.n_pursuers
        n_evaders = self.n_evaders
        if origin:
            # initialize all agents to (0,0)
            allies = agent_utils.create_agents(n_pursuers, mmatrix)
            opponents = agent_utils.create_agents(n_evaders, mmatrix)
        elif random:
            # initialize all agents to random valid positions
            allies = agent_utils.create_agents(n_pursuers, mmatrix, randinit=True)
            opponents = agent_utils.create_agents(n_evaders, mmatrix, randinit=True)
        else:
            if not self.initial_config:
                # if empty do origin
                allies = agent_utils.create_agents(n_pursuers, mmatrix)
                opponents = agent_utils.create_agents(n_evaders, mmatrix)
            else:
            # use the initial configuration
                allies = agent_utils.set_agents(self.initial_config["allies"], mmatrix)
                opponents = agent_utils.set_agents(self.initial_config["opponents"], mmatrix)

        (xs, ys) = mmatrix.shape
        self.ally_layer = AgentLayer(xs, ys, allies)
        self.opponent_layer = AgentLayer(xs, ys, opponents)

        self.n_pursuers = self.ally_layer.n_agents()
        self.n_evaders = self.opponent_layer.n_agents()

        self.ally_actions = np.zeros(self.n_pursuers, dtype=np.int32)
        self.opponent_actions = np.zeros(self.n_evaders, dtype=np.int32)

    def set_agents(self, agent_type):
        if agent_type == "allies":
            self.train_pursuit = True
        else:
            self.train_pursuit = False

    #################################################################


    def collect_obs(self, agent_layer):
        # returns a flattened array of all the observations
        n = agent_layer.n_agents()
        self.local_obs.fill(-0.1) # border walls set to -0.1?
        # loop through agents
        for i in xrange(n):
            # get the obs bounds
            xp, yp = agent_layer.get_position(i)
            
            xlo,xhi,ylo,yhi, xolo,xohi,yolo,yohi = self.obs_clip(xp, yp)

            self.local_obs[i, :, xolo:xohi, yolo:yohi] = self.model_state[0:3, xlo:xhi, ylo:yhi]
        if self.flatten: return self.local_obs.flatten() / self.layer_norm
        return self.local_obs / self.layer_norm
        

    def obs_clip(self, x, y):
        # :( this is a mess, beter way to do the slicing?
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (np.clip(xld, 0, self.xs-1), np.clip(xhd, 0, self.xs-1), np.clip(yld, 0, self.ys-1), np.clip(yhd, 0, self.ys-1))
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(np.clip(yld, -self.obs_offset, 0)) 
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi+1, ylo, yhi+1, xolo, xohi+1, yolo, yohi+1


    def pursuer_reward(self):
        """
        Computes the joint reward for pursuers
        """
        # rewarded for each tagged evader
        ps = self.pursuer_layer.get_state_matrix() # pursuer positions
        es = self.evader_layer.get_state_matrix() # evader positions
        tagged = np.sum((ps > 0) * es) # number of tagged evaders
        rtot = self.catchr * tagged
        return rtot 


    def evader_reward(self):
        """
        Computes the joint reward for evaders
        """
        # penalized for each tagged evader
        ps = self.pursuer_layer.get_state_matrix() # pursuer positions
        es = self.evader_layer.get_state_matrix() # evader positions
        tagged = np.sum((ps > 0) * es) # number of tagged evaders
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
        rems = 0
        for i in xrange(self.evader_layer.n_agents()):
            x,y = self.evader_layer.get_position(i)
            if self.model_state[1,x,y] >= self.n_catch:
                # add prob remove?
                removed_evade.append(i-rems)
                rems += 1
        rems = 0
        for i in xrange(self.pursuer_layer.n_agents()):
            x,y = self.pursuer_layer.get_position(i)
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
        agent_state[x,y] = 1
        self.model_state[0] = self.map_matrix
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state_matrix(), self.ally_layer.get_state_matrix()
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
        agent_state[x,y] = 1
        self.model_state[0] = self.map_matrix
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state_matrix(), self.ally_layer.get_state_matrix()
        self.model_state[3] = agent_state
        return self.model_state

    def transition_pursuer(self, agent_idx, action):
        pursuers = self.ally_layer
        evaders = self.opponent_layer

        pursuer_actions = self.ally_actions
        evader_actions = self.opponent_actions

        # get the pursuer actions
        for i in xrange(self.n_pursuers):
            if i == agent_idx: continue
            state = self.get_layers_pursuer(i)
            pursuer_actions[i] = self.ally_controller.action(state)

        # get evader actions
        for i in xrange(self.n_evaders):
            state = self.get_layers_evader(i)
            evader_actions[i] = self.opponent_controller.action(state)
        
        # evolve the system
        for i in xrange(self.n_pursuers):
            if i == agent_idx: continue
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
            if i == agent_idx: continue
            state = self.get_layers_evader(i)
            evader_actions[i] = self.opponent_controller.action(state)
        
        # evolve the system
        for i in xrange(self.n_pursuers):
            pursuers.move_agent(i, pursuer_actions[i])
        for i in xrange(self.n_evaders):
            if i == agent_idx: continue
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

