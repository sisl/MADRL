#!/usr/bin/env python
#
# File: main.py
#
# Created: Wednesday, July  6 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import sys
sys.path.append('../rltools/')

import numpy as np
import tensorflow as tf

import gym
import rltools.algos
import rltools.log
import rltools.util
from madrl_environments.pursuit.centralized_pursuit_evade import \
    CentralizedPursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from rltools.baseline import LinearFeatureBaseline, MLPBaseline, ZeroBaseline
from rltools.categorical_policy import CategoricalMLPPolicy



SIMPLE_POLICY_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''
TINY_VAL_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"}
    ]
    '''
SIMPLE_VAL_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "relu"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "relu"}
    ]
    '''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)
    parser.add_argument('--max_traj_len', type=int, default=200)
    parser.add_argument('--adaptive_batch', action='store_true', default=False)
    parser.add_argument('--rectangle', type=str, default='10,10')
    parser.add_argument('--n_evaders', type=int, default=5)
    parser.add_argument('--n_pursuers', type=int, default=2)
    parser.add_argument('--obs_range', type=int, default=3)
    parser.add_argument('--n_catch', type=int, default=2)
    parser.add_argument('--policy_hidden_spec', type=str, default=SIMPLE_POLICY_ARCH)
    parser.add_argument('--baseline_type', type=str, default='mlp')
    parser.add_argument('--baseline_hidden_spec', type=str, default=SIMPLE_VAL_ARCH)
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--vf_max_kl', type=float, default=0.01)
    parser.add_argument('--vf_cg_damping', type=float, default=0.01)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)
    args = parser.parse_args()
    env = CentralizedPursuitEvade(TwoDMaps.rectangle_map(*map(int, args.rectangle.split(','))),
                                  n_evaders=args.n_evaders,
                                  n_pursuers=args.n_pursuers,
                                  obs_range=args.obs_range,
                                  n_catch=args.n_catch)
    tboard_dir = '/tmp/madrl_tb'
    policy = CategoricalMLPPolicy(env.observation_space, env.action_space,
                                  hidden_spec=args.policy_hidden_spec,
                                  enable_obsnorm=True,
                                  tblog=tboard_dir, varscope_name='catmlp_policy')

    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(env.observation_space, enable_obsnorm=True)
    elif args.baseline_type == 'mlp':
        baseline = MLPBaseline(env.observation_space, args.baseline_hidden_spec,
                               True, True, max_kl=args.vf_max_kl, damping=args.vf_cg_damping,
                               time_scale=1./args.max_traj_len, varscope_name='mlp_baseline')
    else:
        baseline = ZeroBaseline(env.observation_space)

    step_func = rltools.algos.TRPO(max_kl=args.max_kl)
    popt = rltools.algos.SamplingPolicyOptimizer(
        env=env,
        policy=policy,
        baseline=baseline,
        step_func=step_func,
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        batch_size=32,
        adaptive_batch=args.adaptive_batch
    )
    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    rltools.util.header(argstr)
    log_f = rltools.log.TrainingLog(args.log, [('args', argstr)])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
