from __future__ import absolute_import, print_function

import argparse
import json
import uuid
import datetime
import dateutil.tz
import os.path as osp
import ast

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from madrl_environments.walker.multi_walker import MultiWalkerEnv
from madrl_environments import StandardizedEnv, ObservationBuffer
from rllabwrapper import RLLabEnv

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

# from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
# from rllab.envs.normalized_env import normalize
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
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
    parser.add_argument('--exp_name', type=str, default=default_exp_name,
                        help='Name of the experiment.')

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=1.0)
    parser.add_argument('--reward_scale', type=float, default=1.0)

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler_workers', type=int, default=1)
    parser.add_argument('--max_traj_len', type=int, default=500)
    parser.add_argument('--update_curriculum', action='store_true', default=False)

    parser.add_argument('--n_timesteps', type=int, default=20000)

    parser.add_argument('--control', type=str, default='centralized')

    parser.add_argument('--n_walkers', type=int, default=2)
    parser.add_argument('--reward_mech', type=str, default='local')

    parser.add_argument('--recurrent', type=str, default=None)
    parser.add_argument('--baseline_type', type=str, default='linear')
    parser.add_argument('--policy_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--baseline_hidden_sizes', type=str, default='128,128')

    parser.add_argument('--max_kl', type=float, default=0.01)

    parser.add_argument('--log_dir', type=str, required=False)
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--seed', type=int, help='Random seed for numpy')
    parser.add_argument('--args_data', type=str, help='Pickled data for stub objects')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                        '(all iterations will be saved), "last" (only '
                        'the last iteration will be saved), or "none" '
                        '(do not save snapshots)')
    parser.add_argument(
        '--log_tabular_only', type=ast.literal_eval, default=False,
        help='Whether to only print the tabular log information (in a horizontal format)')

    args = parser.parse_args()

    parallel_sampler.initialize(n_parallel=args.sampler_workers)

    if args.seed is not None:
        set_seed(args.seed)
        parallel_sampler.set_seed(args.seed)

    args.hidden_sizes = tuple(map(int, args.policy_hidden_sizes.split(',')))

    centralized = True if args.control == 'centralized' else False

    env = MultiWalkerEnv(n_walkers=args.n_walkers)

    env = TfEnv(
        RLLabEnv(
            StandardizedEnv(env, scale_reward=args.reward_scale, enable_obsnorm=True),
            mode=args.control))


    if args.recurrent:
        feature_network = MLP(
            name='feature_net',
            input_shape=(env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
            output_dim=4, hidden_sizes=(64, 64), hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None)
        if args.recurrent == 'gru':
            policy = GaussianGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                       hidden_dim=int(args.policy_hidden_sizes), name='policy')
        elif args.recurrent == 'lstm':
            policy = GaussianLSTMPolicy(env_spec=env.spec, feature_network=feature_network,
                                        hidden_dim=int(args.policy_hidden_sizes), name='policy')
    else:
        policy = GaussianMLPPolicy(
            env_spec=env.spec, hidden_sizes=tuple(map(int, args.policy_hidden_sizes.split(','))))

    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif args.baseline_type == 'mlp':
        raise NotImplementedError()
        # baseline = GaussianMLPBaseline(
        #     env_spec=env.spec, hidden_sizes=tuple(map(int, args.baseline_hidden_sizes.split(','))))
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

    """
    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % args.exp_name)
    """

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
