import numpy as np

#################################################################
# Implements multi-agent controllers
#################################################################


class RandomPolicy():

    # constructor
    def __init__(self, n_actions, rng = np.random.RandomState()):
        self.rng = rng
        self.n_actions = n_actions

    def act(self, state):
        return self.rng.randint(self.n_actions) 

