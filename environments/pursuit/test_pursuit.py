from centralized_pursuit_evade import CentralizedPursuitEvade
import gym

from utils import *

import tensorflow as tf

from rltools.algs import TRPOSolver
from rltools.utils import simulate
from rltools.models import softmax_mlp


xs = 10
ys = 10
n_evaders = 1
n_pursuers = 1

map_mat = TwoDMaps.rectangle_map(xs, ys) 

# obs_range should be odd 3, 5, 7, etc
env = CentralizedPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=9, n_catch=1)
#env = gym.make("CartPole-v0")


config = {}
config["train_iterations"] = 100 # number of trpo iterations
config["max_pathlength"] = 150 # maximum length of an env trajecotry
config["gamma"] = 0.95 # discount factor

# lets initialize a model
input_obs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="obs")
net = softmax_mlp(input_obs, env.action_space.n, layers=[32,32], activation=tf.nn.tanh)

solver = TRPOSolver(env, config=config, policy_net=net, input_layer=input_obs)
solver.learn()

simulate(env, solver, 100, render=False)

