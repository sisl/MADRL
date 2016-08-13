#!/usr/bin/env python
#
# File: run_waterworld.py
#
# Created: Thursday, July 14 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json

import numpy as np
import tensorflow as tf

from gym import spaces
import rltools.algos.policyopt
import rltools.log
import rltools.util
from rltools.samplers.serial import SimpleSampler, ImportanceWeightedSampler, DecSampler
from rltools.samplers.parallel import ThreadedSampler, ParallelSampler
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import MAWaterWorld
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

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler', type=str, default='simple')
    parser.add_argument('--sampler_workers', type=int, default=4)
    parser.add_argument('--max_traj_len', type=int, default=500)
    parser.add_argument('--adaptive_batch', action='store_true', default=False)

    parser.add_argument('--n_timesteps', type=int, default=8000)
    parser.add_argument('--n_timesteps_min', type=int, default=1000)
    parser.add_argument('--n_timesteps_max', type=int, default=64000)
    parser.add_argument('--timestep_rate', type=int, default=20)

    parser.add_argument('--is_n_backtrack', type=int, default=1)
    parser.add_argument('--is_randomize_draw', action='store_true', default=False)
    parser.add_argument('--is_n_pretrain', type=int, default=0)
    parser.add_argument('--is_skip_is', action='store_true', default=False)
    parser.add_argument('--is_max_is_ratio', type=float, default=0)

    parser.add_argument('--control', type=str, default='centralized')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--n_evaders', type=int, default=5)
    parser.add_argument('--n_pursuers', type=int, default=3)
    parser.add_argument('--n_poison', type=int, default=10)
    parser.add_argument('--n_coop', type=int, default=2)
    parser.add_argument('--n_sensors', type=int, default=30)
    parser.add_argument('--sensor_range', type=str, default='0.2,0.2,0.2')
    parser.add_argument('--food_reward', type=float, default=3)
    parser.add_argument('--poison_reward', type=float, default=-1)
    parser.add_argument('--encounter_reward', type=float, default=0.05)

    parser.add_argument('--policy_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--baseline_type', type=str, default='mlp')
    parser.add_argument('--baseline_hidden_spec', type=str, default=GAE_ARCH)

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--vf_max_kl', type=float, default=0.01)
    parser.add_argument('--vf_cg_damping', type=float, default=0.01)

    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--log', type=str, required=False)
    parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)

    args = parser.parse_args()

    centralized = True if args.control == 'centralized' else False
    sensor_range = np.array(map(float, args.sensor_range.split(',')))
    assert sensor_range.shape == (args.n_pursuers,)

    env = MAWaterWorld(args.n_pursuers, args.n_evaders, args.n_coop, args.n_poison,
                       n_sensors=args.n_sensors, food_reward=args.food_reward,
                       poison_reward=args.poison_reward, encounter_reward=args.encounter_reward,
                       centralized=centralized, sensor_range=sensor_range, obstacle_loc=None)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if centralized:
        obsfeat_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                                   high=env.agents[0].observation_space.high[0],
                                   shape=(env.agents[0].observation_space.shape[0] *
                                          len(env.agents),))  # XXX
        action_space = spaces.Box(low=env.agents[0].action_space.low[0],
                                  high=env.agents[0].action_space.high[0],
                                  shape=(env.agents[0].action_space.shape[0] *
                                         len(env.agents),))  # XXX
    else:
        obsfeat_space = env.agents[0].observation_space
        action_space = env.agents[0].action_space

    policy = GaussianMLPPolicy(obsfeat_space, action_space, hidden_spec=args.policy_hidden_spec,
                               enable_obsnorm=True, min_stdev=0., init_logstdev=0.,
                               tblog=args.tblog, varscope_name='gaussmlp_policy')
    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(obsfeat_space, enable_obsnorm=True,
                                         varscope_name='pursuit_linear_baseline')
    elif args.baseline_type == 'mlp':
        baseline = MLPBaseline(obsfeat_space, args.baseline_hidden_spec, True, True,
                               max_kl=args.vf_max_kl, damping=args.vf_cg_damping,
                               time_scale=1. / args.max_traj_len,
                               varscope_name='pursuit_mlp_baseline')
    else:
        baseline = ZeroBaseline(obsfeat_space)

    if args.sampler == 'simple':
        if centralized:
            sampler_cls = SimpleSampler
        elif args.control == 'decentralized':
            sampler_cls = DecSampler
        else:
            raise NotImplementedError()
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=args.n_timesteps_min,
                            n_timesteps_max=args.n_timesteps_max, timestep_rate=args.timestep_rate,
                            adaptive=args.adaptive_batch, enable_rewnorm=True)
    elif args.sampler == 'thread':
        sampler_cls = ThreadedSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=args.n_timesteps_min,
                            n_timesteps_max=args.n_timesteps_max, timestep_rate=args.timestep_rate,
                            adaptive=args.adaptive_batch, enable_rewnorm=True)
    elif args.sampler == 'parallel':
        sampler_cls = ParallelSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=args.n_timesteps_min,
                            n_timesteps_max=args.n_timesteps_max, timestep_rate=args.timestep_rate,
                            adaptive=args.adaptive_batch, enable_rewnorm=True,
                            n_workers=args.sampler_workers, mode=args.control)

    elif args.sampler == 'imp':
        sampler_cls = ImportanceWeightedSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=args.n_timesteps_min,
                            n_timesteps_max=args.n_timesteps_max, timestep_rate=args.timestep_rate,
                            adaptive=args.adaptive_batch, enable_rewnorm=True,
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
