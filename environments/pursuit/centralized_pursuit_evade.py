import numpy as np

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
        pursuer_sim: flag indicating if we are simulating pursuers or evaders
        initial_config: dictionary of form
            initial_config['allies']: the initial ally confidguration (matrix)
            initial_config['opponents']: the initial opponent confidguration (matrix)
        """
        
        xs, ys = map_matrix.shape
        self.map_matrix = map_matrix
        
        self.n_evaders = kwargs.pop('n_evaders', 1)
        self.n_pursuers = kwargs.pop('n_pursuers', 1)

        self.obs_range = kwargs.pop('obs_range', 3) # can see 3 grids around them by default

        pursuers = agent_utils.create_agents(self.n_pursuers, map_matrix)
        evaders = agent_utils.create_agents(self.n_evaders, map_matrix)

        self.pursuer_layer = kwargs.pop('ally_layer', AgentLayer(xs, ys, pursuers))
        self.evader_layer = kwargs.pop('opponent_layer', AgentLayer(xs, ys, evaders)) 

        self.low = [0.0 for i in xrange(self.obs_range**2 * self.n_pursuers)]
        self.high = [1.0 for i in xrange(self.obs_range**2 * self.n_pursuers)]

        n_act_purs = self.ally_layer.get_nactions(0)
        n_act_ev = self.opponent_layer.get_nactions(0)

        self.action_space = spaces.Discrete(n_act_purs)
        self.observation_space = spaces.Box([float(self.low), self.high])




        self.ally_controller = kwargs.pop('ally_controller', RandomPolicy(n_act_purs))
        self.opponent_controller = kwargs.pop('opponent_controller', RandomPolicy(n_act_ev)) 

        self.current_agent_layer = np.zeros((xs, ys), dtype=np.int32)

        self.catchr = kwargs.pop('catchr', 1.0)
        self.caughtr = kwargs.pop('caughtr', -1.0)

        self.term_pursuit = kwargs.pop('term_pursui', 10)
        self.term_evade = kwargs.pop('term_pursui', -10)

        self.ally_actions = np.zeros(n_act_purs, dtype=np.int32)
        self.opponent_actions = np.zeros(n_act_ev, dtype=np.int32)

        self.pursuer_sim = kwargs.pop('pursuer_sim', True)

        self.initial_config = kwargs.pop('initial_config', {})

        self.ally_controller = None
        self.opponent_controller = None

        self.model_dims = (4,) + map_matrix.shape

        self.n_actions = n_act_purs

        self.model_state = np.zeros(self.model_dims, dtype=np.float32)

    #################################################################
    # The functions below are the interface with MultiAgentSiulator # 
    #################################################################

    def reset(self):
        self.pursuers = AgentLayer(self.xs, self.ys, 
                                   agent_utils.create_agents(self.n_pursuers, map_matrix, randinit=True))
        self.evaders = AgentLayer(self.xs, self.ys ,
                                  agent_utils.create_agents(self.n_evaders, map_matrix, randinit=True)
        self.model_state[0] = self.map_matrix
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state(), self.ally_layer.get_state()
        self.model_state[3] = agent_state
        return collect_obs(self.pursuers)

    def reward(self):
        r = self.pursuer_reward() if self.pursuer_sim else self.evader_reward()
        return r

    def act(self, agent_idx, action):
        if self.pursuer_sim:
            self.transition_pursuer(agent_idx, action)
        else:
            self.transition_evader(agent_idx, action)

    def observe(self, agent_idx):
        if self.pursuer_sim:
            return self.get_layers_pursuer(agent_idx)
        else:
            return self.get_layers_evader(agent_idx)

    def update_ally_controller(self, controller):
        self.ally_controller = controller

    def update_opponent_controller(self, controller):
        self.opponent_controller = controller

    def is_terminal(self):
        ps = self.ally_layer.get_state() # pursuer positions
        es = self.opponent_layer.get_state() # evader positions
        tagged = np.sum((ps > 0) * es) # number of tagged evaders
        if tagged == self.n_evaders:
            return True
        return False

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
            self.pursuer_sim = True
        else:
            self.pursuer_sim = False

    #################################################################


    def collect_obs(self, agents):
        # returns a flattened array of all the observations



    def pursuer_reward(self):
        """
        Computes the joint reward for pursuers
        """
        # rewarded for each tagged evader
        ps = self.ally_layer.get_state() # pursuer positions
        es = self.opponent_layer.get_state() # evader positions
        tagged = np.sum((ps > 0) * es) # number of tagged evaders
        rtot = self.catchr * tagged
        if tagged == self.n_evaders:
            rtot += self.term_pursuit
        return rtot 

    def evader_reward(self):
        """
        Computes the joint reward for evaders
        """
        # penalized for each tagged evader
        ps = self.ally_layer.get_state() # pursuer positions
        es = self.opponent_layer.get_state() # evader positions
        tagged = np.sum((ps > 0) * es) # number of tagged evaders
        rtot = self.caughtr * tagged
        if tagged == self.n_evaders:
            rtot += self.term_evade
        return rtot 


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
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state(), self.ally_layer.get_state()
        self.model_state[3] = agent_state
        #return (self.map_matrix, self.opponent_layer.get_state(), self.ally_layer.get_state(), agent_state)
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
        self.model_state[1], self.model_state[2] = self.opponent_layer.get_state(), self.ally_layer.get_state()
        self.model_state[3] = agent_state
        #return (self.map_matrix, self.ally_layer.get_state(), self.opponent_layer.get_state(), agent_state)
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

