import os.path as osp

import tensorflow as tf
import rllab.misc.logger as logger
from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.ext import set_seed
from rllab.sampler import parallel_sampler

from rllab.algos.ddpg import DDPG as thDDPG
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy as thDeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction as thContinuousMLPQFunction

from rllab.core.network import MLP as thMLP
from rllab.spaces.box import Box as thBox
from rllab.spaces.discrete import Discrete as thDiscrete
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy as thCategoricalGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy as thCategoricalMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy as thGaussianGRUPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy as thGaussianMLPPolicy

from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy as thDeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction as thContinuousMLPQFunction

from rllabwrapper import RLLabEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import (ConjugateGradientOptimizer,
                                                                      FiniteDifferenceHvp)
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete

from runners import tonamedtuple


def rllab_envpolicy_parser(env, args):
    if isinstance(args, dict):
        args = tonamedtuple(args)

    env = RLLabEnv(env, mode=args.control)
    if args.algo[:2] == 'tf':
        env = TfEnv(env)

        # Policy
        if args.recurrent:
            if args.feature_net:
                feature_network = MLP(name='feature_net', input_shape=(
                    env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
                                      output_dim=args.feature_output,
                                      hidden_sizes=tuple(args.feature_hidden),
                                      hidden_nonlinearity=tf.nn.tanh, output_nonlinearity=None)
            elif args.conv:
                strides = tuple(args.conv_strides)
                chans = tuple(args.conv_channels)
                filts = tuple(args.conv_filters)

                assert len(strides) == len(chans) == len(filts), "strides, chans and filts not equal"
                # only discrete actions supported, should be straightforward to extend to continuous    
                assert isinstance(env.spec.action_space, Discrete), "Only discrete action spaces support conv"
                feature_network = ConvNetwork(
                    name='feature_net',
                    input_shape=env.spec.observation_space.shape,
                    output_dim=args.feature_output,
                    conv_filters=chans,
                    conv_filter_sizes=filts,
                    conv_strides=strides,
                    conv_pads=('VALID',)*len(chans),
                    hidden_sizes=tuple(args.feature_hidden),
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=None)
            else:
                feature_network = None
            if args.recurrent == 'gru':
                if isinstance(env.spec.action_space, Box):
                    policy = GaussianGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                               hidden_dim=int(args.policy_hidden[0]), name='policy')
                elif isinstance(env.spec.action_space, Discrete):
                    policy = CategoricalGRUPolicy(env_spec=env.spec,
                                                  feature_network=feature_network,
                                                  hidden_dim=int(args.policy_hidden[0]),
                                                  name='policy',
                                                  state_include_action=False if args.conv else True)
                else:
                    raise NotImplementedError(env.spec.observation_space)

            elif args.recurrent == 'lstm':
                if isinstance(env.spec.action_space, Box):
                    policy = GaussianLSTMPolicy(env_spec=env.spec, feature_network=feature_network,
                                                hidden_dim=int(args.policy_hidden), name='policy')
                elif isinstance(env.spec.action_space, Discrete):
                    policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                                   feature_network=feature_network,
                                                   hidden_dim=int(args.policy_hidden),
                                                   name='policy')
                else:
                    raise NotImplementedError(env.spec.action_space)

            else:
                raise NotImplementedError(args.recurrent)
        elif args.conv:
            strides = tuple(args.conv_strides)
            chans = tuple(args.conv_channels)
            filts = tuple(args.conv_filters)

            assert len(strides) == len(chans) == len(filts), "strides, chans and filts not equal"
            # only discrete actions supported, should be straightforward to extend to continuous    
            assert isinstance(env.spec.action_space, Discrete), "Only discrete action spaces support conv"
            feature_network = ConvNetwork(
                name='feature_net',
                input_shape=env.spec.observation_space.shape,
                output_dim=env.spec.action_space.n,
                conv_filters=chans,
                conv_filter_sizes=filts,
                conv_strides=strides,
                conv_pads=('VALID',)*len(chans),
                hidden_sizes=tuple(args.policy_hidden),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.softmax)
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, prob_network=feature_network)
        else:
            if isinstance(env.spec.action_space, Box):
                policy = GaussianMLPPolicy(env_spec=env.spec,
                                           hidden_sizes=tuple(args.policy_hidden),
                                           min_std=args.min_std, name='policy')
            elif isinstance(env.spec.action_space, Discrete):
                policy = CategoricalMLPPolicy(env_spec=env.spec,
                                              hidden_sizes=tuple(args.policy_hidden),
                                              name='policy')
            else:
                raise NotImplementedError(env.spec.action_space)
    elif args.algo[:2] == 'th':
        # Policy
        if args.recurrent:
            if args.feature_net:
                feature_network = thMLP(input_shape=(
                    env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim,),
                                        output_dim=args.feature_output,
                                        hidden_sizes=tuple(args.feature_hidden),
                                        hidden_nonlinearity=tf.nn.tanh, output_nonlinearity=None)
            else:
                feature_network = None
            if args.recurrent == 'gru':
                if isinstance(env.spec.observation_space, thBox):
                    policy = thGaussianGRUPolicy(env_spec=env.spec,
                                                 feature_network=feature_network,
                                                 hidden_dim=int(args.policy_hidden[0]),)
                elif isinstance(env.spec.observation_space, thDiscrete):
                    policy = thCategoricalGRUPolicy(env_spec=env.spec,
                                                    feature_network=feature_network,
                                                    hidden_dim=int(args.policy_hidden[0]),)
                else:
                    raise NotImplementedError(env.spec.observation_space)

            # elif args.recurrent == 'lstm':
            #     if isinstance(env.spec.action_space, thBox):
            #         policy = thGaussianLSTMPolicy(env_spec=env.spec,
            #                                       feature_network=feature_network,
            #                                       hidden_dim=int(args.policy_hidden),
            #                                       name='policy')
            #     elif isinstance(env.spec.action_space, thDiscrete):
            #         policy = thCategoricalLSTMPolicy(env_spec=env.spec,
            #                                          feature_network=feature_network,
            #                                          hidden_dim=int(args.policy_hidden),
            #                                          name='policy')
            #     else:
            #         raise NotImplementedError(env.spec.action_space)

            else:
                raise NotImplementedError(args.recurrent)
        else:
            if args.algo == 'thddpg':
                assert isinstance(env.spec.action_space, thBox)
                policy = thDeterministicMLPPolicy(env_spec=env.spec,
                                                  hidden_sizes=tuple(args.policy_hidden),)
            else:
                if isinstance(env.spec.action_space, thBox):
                    policy = thGaussianMLPPolicy(env_spec=env.spec,
                                                 hidden_sizes=tuple(args.policy_hidden),
                                                 min_std=args.min_std)
                elif isinstance(env.spec.action_space, thDiscrete):
                    policy = thCategoricalMLPPolicy(env_spec=env.spec,
                                                    hidden_sizes=tuple(args.policy_hidden),
                                                    min_std=args.min_std)
                else:
                    raise NotImplementedError(env.spec.action_space)

    if args.control == 'concurrent':
        return env, policies
    else:
        return env, policy


class RLLabRunner(object):

    def __init__(self, env, args):
        self.args = args
        # Parallel setup
        parallel_sampler.initialize(n_parallel=args.n_parallel)
        if args.seed is not None:
            set_seed(args.seed)
            parallel_sampler.set_seed(args.seed)

        env, policy = rllab_envpolicy_parser(env, args)

        if not args.algo == 'thddpg':
            # Baseline
            if args.baseline_type == 'linear':
                baseline = LinearFeatureBaseline(env_spec=env.spec)
            elif args.baseline_type == 'zero':
                baseline = ZeroBaseline(env_spec=env.spec)
            else:
                raise NotImplementedError(args.baseline_type)

        # Logger
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

        if args.algo == 'tftrpo':
            self.algo = TRPO(env=env, policy=policy, baseline=baseline, batch_size=args.batch_size,
                             max_path_length=args.max_path_length, n_itr=args.n_iter,
                             discount=args.discount, gae_lambda=args.gae_lambda,
                             step_size=args.step_size, optimizer=ConjugateGradientOptimizer(
                                 hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)) if args.recurrent
                             else None, mode=args.control)
        elif args.algo == 'thddpg':
            qfunc = thContinuousMLPQFunction(env_spec=env.spec)
            if args.exp_strategy == 'ou':
                es = OUStrategy(env_spec=env.spec)
            elif args.exp_strategy == 'gauss':
                es = GaussianStrategy(env_spec=env.spec)
            else:
                raise NotImplementedError()

            self.algo = thDDPG(env=env, policy=policy, qf=qfunc, es=es, batch_size=args.batch_size,
                               max_path_length=args.max_path_length, epoch_length=args.epoch_length,
                               min_pool_size=args.min_pool_size,
                               replay_pool_size=args.replay_pool_size, n_epochs=args.n_iter,
                               discount=args.discount, scale_reward=0.01,
                               qf_learning_rate=args.qfunc_lr, policy_learning_rate=args.policy_lr,
                               eval_samples=args.eval_samples, mode=args.control)

    def __call__(self):
        self.algo.train()
