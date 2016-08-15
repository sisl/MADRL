#!/usr/bin/env python
#
# File: run_con_hostage.py
#
# Created: Sunday, August 14 2016 by rejuvyesh <mail@rejuvyesh.com>
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
from madrl_environments.hostage import ContinuousHostageWorld
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy

from runners.archs import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--interp_alpha', type=float, default=0.5)
    parser.add_argument('--policy_avg_weights', type=str, default='0.33,0.33,0.33')

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

    parser.add_argument('--buffer_size', type=int, default=1)
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
    parser.add_argument('--min_std', type=float, default=0)

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

    policy_avg_weights = np.array(map(float, args.policy_avg_weights.split(',')))
    assert len(policy_avg_weights) == args.n_good

    env = ContinuousHostageWorld(args.n_good, args.n_hostage, args.n_bad, args.n_coop_save,
                                 args.n_coop_avoid, n_sensors=args.n_sensors,
                                 sensor_range=args.sensor_range, save_reward=args.save_reward,
                                 hit_reward=args.hit_reward, encounter_reward=args.encounter_reward,
                                 bomb_reward=args.bomb_reward)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    policies = [GaussianMLPPolicy(agent.observation_space, agent.action_space,
                                  hidden_spec=args.policy_hidden_spec, enable_obsnorm=True,
                                  min_stdev=args.min_std, init_logstdev=0., tblog=args.tblog,
                                  varscope_name='gaussmlp_policy_{}'.format(agid))
                for agid, agent in enumerate(env.agents)]
    if args.blend_freq:
        assert all(
            [agent.observation_space == env.agents[0].observation_space for agent in env.agents])
        target_policy = GaussianMLPPolicy(env.agents[0].observation_space,
                                          env.agents[0].action_space,
                                          hidden_spec=args.policy_hidden_spec, enable_obsnorm=True,
                                          min_stdev=0., init_logstdev=0., tblog=args.tblog,
                                          varscope_name='targetgaussmlp_policy')
    else:
        target_policy = None

    if args.baseline_type == 'linear':
        baselines = [LinearFeatureBaseline(agent.observation_space, enable_obsnorm=True,
                                           varscope_name='linear_baseline_{}'.format(agid))
                     for agid, agent in enumerate(env.agents)]
    elif args.baseline_type == 'mlp':
        baselines = [MLPBaseline(agent.observation_space, args.baseline_hidden_spec,
                                 enable_obsnorm=True, enable_vnorm=True, max_kl=args.vf_max_kl,
                                 damping=args.vf_cg_damping, time_scale=1. / args.max_traj_len,
                                 varscope_name='mlp_baseline_{}'.format(agid))
                     for agid, agent in enumerate(env.agents)]
    else:
        baselines = [ZeroBaseline(agent.observation_space) for agent in env.agents]

    if args.sampler == 'parallel':
        sampler_cls = ParallelSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=args.n_timesteps_min,
                            n_timesteps_max=args.n_timesteps_max, timestep_rate=args.timestep_rate,
                            adaptive=args.adaptive_batch, enable_rewnorm=True,
                            n_workers=args.sampler_workers, mode='concurrent')

    else:
        raise NotImplementedError()

    step_func = rltools.algos.policyopt.TRPO(max_kl=args.max_kl)

    popt = rltools.algos.policyopt.ConcurrentPolicyOptimizer(
        env=env, policies=policies, baselines=baselines, step_func=step_func,
        discount=args.discount, gae_lambda=args.gae_lambda, sampler_cls=sampler_cls,
        sampler_args=sampler_args, n_iter=args.n_iter, target_policy=target_policy,
        weights=policy_avg_weights, interp_alpha=args.interp_alpha)

    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    rltools.util.header(argstr)
    log_f = rltools.log.TrainingLog(args.log, [('args', argstr)], debug=args.debug)

    blend_freq = 20  # XXX
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f, blend_freq, args.save_freq)


if __name__ == '__main__':
    main()
