# evalaute against stationary evaders

from madrl_environments import CentralizedPursuitEvade
from madrl_environments.pursuit import TwoDMaps
from madrl_environments.pursuit import SingleActionPolicy

from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


from rltools.algs import TRPOSolver
from rltools.utils import evaluate_time
from rltools.utils import simulate
from rltools.models import softmax_mlp

xs = 10
ys = 10
n_evaders = 1
n_pursuers = 2

map_mat = TwoDMaps.rectangle_map(xs, ys) 

evader_policy = SingleActionPolicy(4) # stationary action

runs = [4, 1, 3, 1]
ranges = [3, 5, 7, 9]
oranges = np.array([ranges for r in runs]).flatten()
model_paths = [join("../data/obs_range_sweep_2layer", "obs_range_"+str(o), "run"+str(r), "final_model.ckpt") for o, r in zip(oranges, runs)]

n_traj = 200
max_steps = 500

results = np.zeros((len(ranges), 2, 3))

count = 0
for ran, path in zip(ranges, model_paths):
    print path

    env = CentralizedPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=ran, n_catch=2,
            evader_controller=evader_policy)

    input_obs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="obs")
    net = softmax_mlp(input_obs, env.action_space.n, layers=[128,128], activation=tf.nn.tanh)
    solver = TRPOSolver(env, policy_net=net, input_layer=input_obs)
    solver.load(path)

    solver.train = True # stochastic policy
    r, t = evaluate_time(env, solver, max_steps, n_traj) 
    results[count, 0] = [n_traj, t.mean(), t.std()]
    solver.train = False # deterministic policy
    r, t = evaluate_time(env, solver, max_steps, n_traj) 
    results[count, 1] = [n_traj, t.mean(), t.std()]

    count += 1
    tf.reset_default_graph()

m = results[:,0,1]
sig = results[:,0,2] / np.sqrt(n_traj)
plt.plot(ranges, m, 'k', color='#CC4F1B', label='Stochastic Policy')
plt.fill_between(ranges, m-sig, m+sig, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

m = results[:,1,1]
sig = results[:,1,2] / np.sqrt(n_traj)
plt.plot(ranges, m, 'k', color='#1B2ACC', label='Deterministic Policy')
plt.fill_between(ranges, m-sig, m+sig, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)

#plt.ylim([3,5.5])

plt.xlabel('Observation Ranges')
plt.ylabel('Time to Find Evader')
plt.title('Centralized Pursuit Policy with Stationay Evader')

plt.ylim([0,160])
plt.grid()
plt.legend()
plt.savefig('results/stationary_eval/two_layer_centralized_eval_best.pdf') 
