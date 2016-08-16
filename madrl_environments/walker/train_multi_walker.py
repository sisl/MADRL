#!/usr/bin/env python
#
# File: simple_continuous.py
#
# Created: Thursday, July 14 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import sys
import pprint
sys.path.append('../rltools/')

import numpy as np
import tensorflow as tf
import h5py

import gym
import rltools.algos.policyopt
import rltools.log
import rltools.util
from rltools.samplers.serial import SimpleSampler, ImportanceWeightedSampler, DecSampler
from rltools.samplers.parallel import ThreadedSampler
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy

from multi_walker import MultiWalkerEnv


GAE_ARCH = '''[
        {"type": "fc", "n": 200},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 100},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 50},
        {"type": "nonlin", "func": "tanh"}
]
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler', type=str, default='simple')
    parser.add_argument('--max_traj_len', type=int, default=500)
    parser.add_argument('--n_timesteps', type=int, default=8000)  # number of traj in an iteration

    parser.add_argument('--n_workers', type=int, default=4)  # number of parallel workers for sampling

    parser.add_argument('--policy_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--baseline_type', type=str, default='mlp')
    parser.add_argument('--baseline_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--vf_max_kl', type=float, default=0.01)
    parser.add_argument('--vf_cg_damping', type=float, default=0.01)

    parser.add_argument('--n_walkers', type=int, default=2)

    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)
    parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')

    parser.add_argument('--load_checkpoint', type=str, default='none') 

    parser.set_defaults(debug=True)

    args = parser.parse_args()

    env = MultiWalkerEnv(n_walkers=args.n_walkers)

    if args.load_checkpoint is not 'none':
        filename, file_key = rltools.util.split_h5_name(args.load_checkpoint)
        print('Loading parameters from {} in {}'.format(file_key, filename))
        with h5py.File(filename, 'r') as f:
            train_args = json.loads(f.attrs['args'])
            dset = f[file_key]

            pprint.pprint(dict(dset.attrs))
        policy = GaussianMLPPolicy(env.observation_space, env.action_space,
                                   hidden_spec=train_args['policy_hidden_spec'], enable_obsnorm=True,
                                   min_stdev=0., init_logstdev=0., tblog=train_args['tblog'],
                                   varscope_name='gaussmlp_policy')
    else:

        policy = GaussianMLPPolicy(env.observation_space, env.action_space,
                                   hidden_spec=args.policy_hidden_spec, enable_obsnorm=True,
                                   min_stdev=0., init_logstdev=0., tblog=args.tblog,
                                   varscope_name='gaussmlp_policy')
    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(env.observation_space, enable_obsnorm=True,
                                         varscope_name='pursuit_linear_baseline')
    elif args.baseline_type == 'mlp':
        baseline = MLPBaseline(env.observation_space, args.baseline_hidden_spec, True, True,
                               max_kl=args.vf_max_kl, damping=args.vf_cg_damping,
                               time_scale=1. / args.max_traj_len,
                               varscope_name='pursuit_mlp_baseline')
    else:
        baseline = ZeroBaseline(env.observation_space)

    if args.sampler == 'simple':
        sampler_cls = DecSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=4000, n_timesteps_max=64000, timestep_rate=40,
                            adaptive=False)
    elif args.sampler == 'parallel':
        sampler_cls = ParallelSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=4000, n_timesteps_max=64000, timestep_rate=40,
                            adaptive=False, n_workers=args.sampler_workers)
    else:
        raise NotImplementedError()
    step_func = rltools.algos.policyopt.TRPO(max_kl=args.max_kl)
    popt = rltools.algos.policyopt.SamplingPolicyOptimizer(env=env, policy=policy,
                                                           baseline=baseline, step_func=step_func,
                                                           discount=args.discount,
                                                           gae_lambda=args.gae_lambda,
                                                           sampler_cls=sampler_cls,
                                                           sampler_args=sampler_args,
                                                           n_iter=args.n_iter)
    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    rltools.util.header(argstr)
    log_f = rltools.log.TrainingLog(args.log, [('args', argstr)], debug=args.debug)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if args.load_checkpoint is not 'none':
            policy.load_h5(sess, filename, file_key)
        popt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
