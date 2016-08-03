#!/usr/bin/env python
#
# File: run_hostage.py
#
# Created: Monday, August  1 2016 by rejuvyesh <mail@rejuvyesh.com>
# License: GNU GPL 3 <http://www.gnu.org/copyleft/gpl.html>
#
from __future__ import absolute_import, print_function

import argparse
import json
import sys
import uuid
sys.path.append('../rltools/')

import numpy as np
import tensorflow as tf

import gym
import rltools.algos.policyopt
import rltools.log
import rltools.util
from rltools.samplers.serial import SimpleSampler, ImportanceWeightedSampler, DecSampler
from rltools.samplers.parallel import ThreadedSampler, ParallelSampler
from madrl_environments.hostage import ContinuousHostageWorld
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy

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
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

GAE_TYPE_VAL_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 64},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
]
'''

GAE_ARCH = '''[
        {"type": "fc", "n": 100},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 50},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 25},
        {"type": "nonlin", "func": "tanh"}
]
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--sampler', type=str, default='simple')
    parser.add_argument('--sampler_workers', type=int, default=2)
    parser.add_argument('--max_traj_len', type=int, default=1500)
    parser.add_argument('--adaptive_batch', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--min_batch_size', type=int, default=4)
    parser.add_argument('--max_batch_size', type=int, default=64)
    parser.add_argument('--batch_rate', type=int, default=40)
    parser.add_argument('--is_n_backtrack', type=int, default=1)
    parser.add_argument('--is_randomize_draw', action='store_true', default=False)
    parser.add_argument('--is_n_pretrain', type=int, default=0)
    parser.add_argument('--is_skip_is', action='store_true', default=False)
    parser.add_argument('--is_max_is_ratio', type=float, default=0)

    parser.add_argument('--control', type=str, default='centralized')
    parser.add_argument('--n_good', type=int, default=3)
    parser.add_argument('--n_hostage', type=int, default=5)
    parser.add_argument('--n_bad', type=int, default=5)
    parser.add_argument('--n_coop_save', type=int, default=2)
    parser.add_argument('--n_coop_avoid', type=int, default=2)
    parser.add_argument('--n_sensors', type=int, default=20)
    parser.add_argument('--sensor_range', type=float, default=0.2)
    parser.add_argument('--save_reward', type=float, default=3)
    parser.add_argument('--hit_reward', type=float, default=-1)
    parser.add_argument('--encounter_reward', type=float, default=0.01)
    parser.add_argument('--bomb_reward', type=float, default=-10.)

    parser.add_argument('--policy_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--baseline_type', type=str, default='mlp')
    parser.add_argument('--baseline_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--vf_max_kl', type=float, default=0.01)
    parser.add_argument('--vf_cg_damping', type=float, default=0.01)

    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)
    parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb_{}'.format(uuid.uuid4()))
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)

    args = parser.parse_args()

    centralized = True if args.control == 'centralized' else False

    env = ContinuousHostageWorld(args.n_good, args.n_hostage, args.n_bad, args.n_coop_save,
                                 args.n_coop_avoid, n_sensors=args.n_sensors,
                                 sensor_range=args.sensor_range, save_reward=args.save_reward,
                                 hit_reward=args.hit_reward, encounter_reward=args.encounter_reward,
                                 bomb_reward=args.bomb_reward, centralized=centralized)

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
        if centralized:
            sampler_cls = SimpleSampler
        elif args.control == 'decentralized':
            sampler_cls = DecSampler
        else:
            raise NotImplementedError()
        sampler_args = dict(max_traj_len=args.max_traj_len, batch_size=args.batch_size,
                            min_batch_size=args.min_batch_size, max_batch_size=args.max_batch_size,
                            batch_rate=args.batch_rate, adaptive=args.adaptive_batch)
    elif args.sampler == 'thread':
        sampler_cls = ThreadedSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, batch_size=args.batch_size,
                            min_batch_size=args.min_batch_size, max_batch_size=args.max_batch_size,
                            batch_rate=args.batch_rate, adaptive=args.adaptive_batch)
    elif args.sampler == 'parallel':
        sampler_cls = ParallelSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, batch_size=args.batch_size,
                            min_batch_size=args.min_batch_size, max_batch_size=args.max_batch_size,
                            batch_rate=args.batch_rate, adaptive=args.adaptive_batch,
                            n_workers=args.sampler_workers, mode=args.control)
    elif args.sampler == 'imp':
        sampler_cls = ImportanceWeightedSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, batch_size=args.batch_size,
                            min_batch_size=args.min_batch_size, max_batch_size=args.max_batch_size,
                            batch_rate=args.batch_rate, adaptive=args.adaptive_batch,
                            n_backtrack=args.is_n_backtrack, randomize_draw=args.is_randomize_draw,
                            n_pretrain=args.is_n_pretrain, skip_is=args.is_skip_is,
                            max_is_ratio=args.is_max_is_ratio)
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
        popt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
