from centralized_pursuit_evade import CentralizedPursuitEvade
import gym

from utils import *

import tensorflow as tf

from rltools.algs import TRPOSolver
from rltools.utils import simulate
from rltools.models import softmax_mlp


xs = 10
ys = 10
n_evaders = 5
n_pursuers = 2

map_mat = TwoDMaps.rectangle_map(xs, ys) 

# obs_range should be odd 3, 5, 7, etc
#env = CentralizedPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=9, n_catch=2)


config = {}
config["train_iterations"] = 1000 # number of trpo iterations
config["max_pathlength"] = 250 # maximum length of an env trajecotry
config["timesteps_per_batch"] = 1000
config["eval_trajectories"] = 50
config["eval_every"] = 50
config["gamma"] = 0.95 # discount factor

#oranges = [3,5,7]
#opaths = ["data/obs_range_sweep/obs_range_3", "data/obs_range_sweep/obs_range_5", "data/obs_range_sweep/obs_range_7"]

oranges = [3, 3, 3, 3]
opaths = ["data/obs_range_sweep/obs_range_3/run2", "data/obs_range_sweep/obs_range_3/run3",
"data/obs_range_sweep/obs_range_3/run4", "data/obs_range_sweep/obs_range_3/run5"]

for ran, path in zip(oranges, opaths):
    # obs_range should be odd 3, 5, 7, etc
    env = CentralizedPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=ran, n_catch=2)

    config["save_path"] = path

    # lets initialize a model
    input_obs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="obs" + str(ran))
    net = softmax_mlp(input_obs, env.action_space.n, layers=[128,128], activation=tf.nn.tanh)

    solver = TRPOSolver(env, config=config, policy_net=net, input_layer=input_obs)
    solver.learn()
    tf.reset_default_graph()

simulate(env, solver, 100, render=False)
