from dec_pursuit_evade import DecPursuitEvade
import gym

from utils import *

import tensorflow as tf

xs = 10
ys = 10
n_evaders = 5
n_pursuers = 4

map_mat = TwoDMaps.rectangle_map(xs, ys) 

# obs_range should be odd 3, 5, 7, etc
env = DecPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=9, n_catch=2, surround=True)

o = env.reset()

a = [4,4,4,4]

env.evader_layer.set_position(0, 8, 1)
env.evader_layer.set_position(1, 8, 1)
env.evader_layer.set_position(2, 8, 1)
env.evader_layer.set_position(3, 8, 1)
env.evader_layer.set_position(4, 8, 1)

env.pursuer_layer.set_position(0, 7, 1)
env.pursuer_layer.set_position(1, 8, 0)
env.pursuer_layer.set_position(2, 9, 1)
env.pursuer_layer.set_position(3, 8, 2)

o, r, done, _ = env.step(a)
