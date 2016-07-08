# compare each policy (including fully obs)

from madrl_environments import CentralizedPursuitEvade
from madrl_environments.pursuit import TwoDMaps

from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


from rltools.algs import TRPOSolver
from rltools.utils import evaluate
from rltools.utils import simulate
from rltools.models import softmax_mlp

#################################################################
######################## PARAMS #################################
#################################################################
layers = [128,64,64] # or [128,128]
model_path = "../data/obs_range_sweep_3layer"
plot_title = "Evaluation of Three Layer Centralized Controller"
save_path = "results/three_layer_centralized_eval.pdf"


xs = 10
ys = 10
n_evaders = 5
n_pursuers = 2

map_mat = TwoDMaps.rectangle_map(xs, ys) 

runs = [1]
ranges = [3, 5, 7, 9]
oranges = np.array([ranges for r in runs]).flatten()
model_paths = [join("../data/obs_range_sweep_2layer_random_evaders_urgency", "obs_range_"+str(o), "run"+str(r), "final_model.ckpt") for o in oranges for r in runs]

n_traj = 100
max_steps = 250

results = np.zeros((len(ranges), len(runs), 2, 3))


best_stds_stoch = np.zeros(len(ranges)) 
best_means_stoch = np.zeros(len(ranges)) 
best_stds_det = np.zeros(len(ranges)) 
best_means_det = np.zeros(len(ranges)) 
pidx = 0
for i, ran in enumerate(ranges):
    for j in xrange(len(runs)):
        print model_paths[pidx]

        env = CentralizedPursuitEvade(map_mat, n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=ran, n_catch=2)

        input_obs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="obs")
        net = softmax_mlp(input_obs, env.action_space.n, layers=[128,128], activation=tf.nn.tanh)
        solver = TRPOSolver(env, policy_net=net, input_layer=input_obs)
        solver.load(model_paths[pidx])

        solver.train = True # stochastic policy
        r = evaluate(env, solver, max_steps, n_traj) 
        smean, sstd = r.mean(), r.std()
        results[i, j, 0] = [n_traj, r.mean(), r.std()]

        solver.train = False # deterministic policy
        r = evaluate(env, solver, max_steps, n_traj) 
        dmean, dstd = r.mean(), r.std()
        results[i, j, 1] = [n_traj, r.mean(), r.std()]
        if smean > best_means_stoch[i]:
            best_means_stoch[i] = smean
            best_stds_stoch[i] = sstd
            best_means_det[i] = dmean
            best_stds_det[i] = dstd

        pidx += 1
        tf.reset_default_graph()


plt.figure(1)

plt.plot(ranges, best_means_stoch, 'k', color='#CC4F1B', label='Stochastic Policy')
plt.fill_between(ranges, best_means_stoch-best_stds_stoch, best_means_stoch+best_stds_stoch, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(ranges, best_means_det, 'k', color='#1B2ACC', label='Deterministic Policy')
plt.fill_between(ranges, best_means_det-best_stds_det, best_means_det+best_stds_det, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)

#plt.ylim([3,5.5])

plt.xlabel('Observation Ranges')
plt.ylabel('Average Rewards')
plt.title('Centralized Pursuit Policy Evaluation')
plt.grid()
plt.legend(loc=4)
plt.savefig('results/policy_eval/two_layer_centralized_eval_random_evaders_urgency_best.pdf') 

"""
stoch_mean = []
stoch_best = []
stoch_worst = []
stoch_std = []
det_mean = []
det_best = []
det_worst = []
det_std = []
for i in xrange(len(ranges)):
    stoch_mean.append(results[i,:,0,1].mean())
    stoch_std.append(results[i,:,0,1].std()/np.sqrt(n_traj))
    stoch_best.append(results[i,:,0,1].max())
    stoch_worst.append(results[i,:,0,1].min())
    det_mean.append(results[i,:,1,1].mean())
    det_std.append(results[i,:,1,1].std()/np.sqrt(n_traj))
    det_best.append(results[i,:,1,1].max())
    det_worst.append(results[i,:,1,1].min())

stoch_mean = np.array(stoch_mean)
stoch_std = np.array(stoch_std)
det_mean = np.array(det_mean)
det_std = np.array(det_std)

plt.plot(ranges, stoch_mean, 'k', color='#CC4F1B', label='Stochastic Policy')
plt.fill_between(ranges, stoch_mean-stoch_std, stoch_mean+stoch_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(ranges, det_mean, 'k', color='#1B2ACC', label='Deterministic Policy')
plt.fill_between(ranges, det_mean-det_std, det_mean+det_std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)

plt.ylim([3,5.5])

plt.xlabel('Observation Ranges')
plt.ylabel('Average Rewards')
plt.title('Evaluation of Three Layer Centralized Controller')
plt.grid()
plt.legend()

plt.savefig('results/three_layer_centralized_eval.pdf')
"""
