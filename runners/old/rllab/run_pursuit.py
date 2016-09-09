#!/usr/bin/env python
#
# File: run_pursuit.py
#
# Created: Wednesday, July  6 2016 by megorov <mail@megorov.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import uuid
import datetime
import dateutil.tz
import os.path as osp
import ast

import joblib

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from madrl_environments import StandardizedEnv
from rllabwrapper import RLLabEnv

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.sampler import parallel_sampler
import rllab.misc.logger as logger
from rllab.misc.ext import set_seed
from rllab import config

def main():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=1.0)
    parser.add_argument('--reward_scale', type=float, default=1.0)

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler_workers', type=int, default=1)
    parser.add_argument('--max_traj_len', type=int, default=250)
    parser.add_argument('--update_curriculum', action='store_true', default=False)
    parser.add_argument('--n_timesteps', type=int, default=8000)
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
    parser.add_argument('--map_file', type=str, default='../maps/map_pool.npy')
    parser.add_argument('--flatten', action='store_true', default=False)
    parser.add_argument('--reward_mech', type=str, default='global')
    parser.add_argument('--catchr', type=float, default=0.1)
    parser.add_argument('--term_pursuit', type=float, default=5.0)

    parser.add_argument('--recurrent', type=str, default=None)
    parser.add_argument('--policy_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--baselin_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--baseline_type', type=str, default='linear')

    parser.add_argument('--conv', action='store_true', default=False)

    parser.add_argument('--max_kl', type=float, default=0.01)

    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--log_dir', type=str, required=False)
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=False,
                        help='Whether to only print the tabular log information (in a horizontal format)')


    args = parser.parse_args()

    parallel_sampler.initialize(n_parallel=args.sampler_workers)

    if args.seed is not None:
        set_seed(args.seed)
        parallel_sampler.set_seed(args.seed)

    args.hidden_sizes = tuple(map(int, args.policy_hidden_sizes.split(',')))


    if args.checkpoint:
       with tf.Session() as sess:
        data = joblib.load(args.checkpoint)
        policy = data['policy']
        env = data['env']
    else: 
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
                           flatten=args.flatten,
                           reward_mech=args.reward_mech,
                           catchr=args.catchr,
                           term_pursuit=args.term_pursuit)

        env = TfEnv(
            RLLabEnv(
                StandardizedEnv(env, scale_reward=args.reward_scale, enable_obsnorm=False),
                mode=args.control))

        if args.recurrent:
            if args.conv:
                feature_network = ConvNetwork(
                    name='feature_net',
                    input_shape=emv.spec.observation_space.shape,
                    output_dim=5, 
                    conv_filters=(16,32,32),
                    conv_filter_sizes=(3,3,3),
                    conv_strides=(1,1,1),
                    conv_pads=('VALID','VALID','VALID'),
                    hidden_sizes=(64,), 
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=tf.nn.softmax)
            else:
                feature_network = MLP(
                    name='feature_net',
                    input_shape=(env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
                    output_dim=5, hidden_sizes=(256,128,64), hidden_nonlinearity=tf.nn.tanh,
                    output_nonlinearity=None)
            if args.recurrent == 'gru':
                policy = CategoricalGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                           hidden_dim=int(args.policy_hidden_sizes), name='policy')
            elif args.recurrent == 'lstm':
                policy = CategoricalLSTMPolicy(env_spec=env.spec, feature_network=feature_network,
                                            hidden_dim=int(args.policy_hidden_sizes), name='policy')
        elif args.conv:
            feature_network = ConvNetwork(
                name='feature_net',
                input_shape=env.spec.observation_space.shape,
                output_dim=5, 
                conv_filters=(8,16),
                conv_filter_sizes=(3,3),
                conv_strides=(2,1),
                conv_pads=('VALID','VALID'),
                hidden_sizes=(32,), 
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.softmax)
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, prob_network=feature_network)
        else:
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=args.hidden_sizes)

    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    else:
        baseline = ZeroBaseline(env_spec=env.spec)

    # logger
    default_log_dir = config.LOG_DIR
    if args.log_dir is None:
        log_dir = osp.join(default_log_dir, args.exp_name)
    else:
        log_dir = args.log_dir
    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % args.exp_name)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.n_timesteps,
        max_path_length=args.max_traj_len,
        n_itr=args.n_iter,
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        step_size=args.max_kl,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)) if
        args.recurrent else None,
        mode=args.control,)

    algo.train()


if __name__ == '__main__':
    main()
