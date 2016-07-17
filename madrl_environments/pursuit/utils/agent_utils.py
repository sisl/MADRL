import numpy as np

from six.moves import xrange

from .TwoDAgent import TwoDAgent

#################################################################
# Implements utility functions for multi-agent DRL
#################################################################


def create_agents(nagents, map_matrix, randinit=False):
    """
    Initializes the agents on a map (map_matrix)
    -nagents: the number of agents to put on the map
    -randinit: if True will place agents in random, feasible locations
               if False will place all agents at 0
    """
    xs, ys = map_matrix.shape
    agents = []
    for i in xrange(nagents):
        xinit, yinit = (0, 0)
        if randinit:
            xinit, yinit = feasible_position(map_matrix)
        agent = TwoDAgent(xs, ys, map_matrix)
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents


def feasible_position(map_matrix):
    """
    Returns a feasible position on map (map_matrix)
    """
    xs, ys = map_matrix.shape
    loop_count = 0
    while True:
        x = np.random.randint(xs)
        y = np.random.randint(ys)
        if map_matrix[x, y] != -1:
            return (x, y)


def set_agents(agent_matrix, map_matrix):
    # check input sizes
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError("Agent configuration and map matrix have mis-matched sizes")

    agents = []
    xs, ys = agent_matrix.shape
    for i in xrange(xs):
        for j in xrange(ys):
            n_agents = agent_matrix[i, j]
            if n_agents > 0:
                if map_matrix[i, j] == -1:
                    raise ValueError(
                        "Trying to place an agent into a building: check map matrix and agent configuration")
                agent = TwoDAgent(xs, ys, map_matrix)
                agent.set_position(i, j)
                agents.append(agent)
    return agents
