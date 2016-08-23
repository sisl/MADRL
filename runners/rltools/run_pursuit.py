#!/usr/bin/env python
#
# File: run_pursuit.py
#
# Created: Wednesday, July  6 2016 by megorov <mail@megorov.com>
#
from __future__ import absolute_import, print_function

import argparse
import json

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

import rltools.algos.policyopt
import rltools.log
import rltools.util
from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from pursuit_policy import PursuitCentralMLPPolicy
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.categorical import CategoricalMLPPolicy
from rltools.samplers.parallel import ParallelSampler
from rltools.samplers.serial import DecSampler, SimpleSampler
from runners import get_arch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler', type=str, default='simple')
    parser.add_argument('--sampler_workers', type=int, default=4)
    parser.add_argument('--max_traj_len', type=int, default=250)
    parser.add_argument('--adaptive_batch', action='store_true', default=False)
    parser.add_argument('--update_curriculum', action='store_true', default=False)

    parser.add_argument('--n_timesteps', type=int, default=8000)
    parser.add_argument('--n_timesteps_min', type=int, default=1000)
    parser.add_argument('--n_timesteps_max', type=int, default=64000)
    parser.add_argument('--timestep_rate', type=int, default=20)

    parser.add_argument('--control', type=str, default='centralized')
    parser.add_argument('--rectangle', type=str, default='10,10')
    parser.add_argument('--map_type', type=str, default='rectangle')
    parser.add_argument('--n_evaders', type=int, default=5)
    parser.add_argument('--n_pursuers', type=int, default=2)
    parser.add_argument('--obs_range', type=int, default=3)
    parser.add_argument('--n_catch', type=int, default=2)
    parser.add_argument('--urgency', type=float, default=0.0)
    parser.add_argument('--pursuit', dest='train_pursuit', action='store_true')
    parser.add_argument('--evade', dest='train_pursuit', action='store_false')
    parser.set_defaults(train_pursuit=True)
    parser.add_argument('--surround', action='store_true', default=False)
    parser.add_argument('--constraint_window', type=float, default=1.0)
    parser.add_argument('--sample_maps', action='store_true', default=False)
    parser.add_argument('--map_file', type=str, default='maps/map_pool.npy')
    parser.add_argument('--flatten', action='store_true', default=False)

    parser.add_argument('--policy_hidden_spec', type=str, default='SIMPLE_CONV_ARCH')

    parser.add_argument('--baseline_type', type=str, default='mlp')
    parser.add_argument('--baseline_hidden_spec', type=str, default='SIMPLE_CONV_ARCH')

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

    args.policy_hidden_spec = get_arch(args.policy_hidden_spec)
    args.baseline_hidden_spec = get_arch(args.baseline_hidden_spec)

    if args.sample_maps:
        map_pool = np.load(args.map_file)
    else:
        if args.map_type == 'rectangle':
            env_map = TwoDMaps.rectangle_map(*map(int, args.rectangle.split(',')))
        elif args.map_type == 'complex':
            env_map = TwoDMaps.complex_map(*map(int, args.rectangle.split(',')))
        else:
            raise NotImplementedError()
        map_pool = [env_map]
    env = PursuitEvade(map_pool, n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
                       obs_range=args.obs_range, n_catch=args.n_catch,
                       train_pursuit=args.train_pursuit, urgency_reward=args.urgency,
                       surround=args.surround, sample_maps=args.sample_maps,
                       constraint_window=args.constraint_window,
                       flatten=args.flatten)

    if args.control == 'centralized':
        obsfeat_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                                   high=env.agents[0].observation_space.high[0],
                                   shape=(env.agents[0].observation_space.shape[0] *
                                          len(env.agents),))  # XXX
        action_space = spaces.Discrete(env.agents[0].action_space.n * len(env.agents))

    elif args.control == 'decentralized':
        obsfeat_space = env.agents[0].observation_space
        action_space = env.agents[0].action_space
        env.reward_mech = 'local'
    else:
        raise NotImplementedError()

    policy = CategoricalMLPPolicy(obsfeat_space, action_space, hidden_spec=args.policy_hidden_spec,
                                  enable_obsnorm=True, tblog=args.tblog,
                                  varscope_name='pursuit_catmlp_policy')

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
        if args.control == "centralized":
            sampler_cls = SimpleSampler
        elif args.control == "decentralized":
            sampler_cls = DecSampler
        else:
            raise NotImplementedError()
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
    else:
        raise NotImplementedError()
    step_func = rltools.algos.policyopt.TRPO(max_kl=args.max_kl)
    popt = rltools.algos.policyopt.SamplingPolicyOptimizer(env=env, policy=policy,
                                                           baseline=baseline, step_func=step_func,
                                                           discount=args.discount,
                                                           gae_lambda=args.gae_lambda,
                                                           sampler_cls=sampler_cls,
                                                           sampler_args=sampler_args,
                                                           n_iter=args.n_iter,
                                                           update_curriculum=args.update_curriculum)
    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    rltools.util.header(argstr)
    log_f = rltools.log.TrainingLog(args.log, [('args', argstr)], debug=args.debug)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
