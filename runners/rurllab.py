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
from sandbox.rocky.tf.algos.ma_trpo import MATRPO
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from sandbox.rocky.tf.envs.base import MATfEnv
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

    env = RLLabEnv(env, ma_mode=args.control)
    if args.algo[:2] == 'tf':
        env = MATfEnv(env)

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

                assert len(strides) == len(chans) == len(
                    filts), "strides, chans and filts not equal"
                # only discrete actions supported, should be straightforward to extend to continuous
                assert isinstance(env.spec.action_space,
                                  Discrete), "Only discrete action spaces support conv"
                feature_network = ConvNetwork(name='feature_net',
                                              input_shape=env.spec.observation_space.shape,
                                              output_dim=args.feature_output, conv_filters=chans,
                                              conv_filter_sizes=filts, conv_strides=strides,
                                              conv_pads=('VALID',) * len(chans),
                                              hidden_sizes=tuple(args.feature_hidden),
                                              hidden_nonlinearity=tf.nn.relu,
                                              output_nonlinearity=None)
            else:
                feature_network = None
            if args.recurrent == 'gru':
                if isinstance(env.spec.action_space, Box):
                    if args.control == 'concurrent':
                        policies = [
                            GaussianGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                              hidden_dim=int(args.policy_hidden[0]),
                                              name='policy_{}'.format(agid))
                            for agid in range(len(env.agents))
                        ]
                    policy = GaussianGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                               hidden_dim=int(args.policy_hidden[0]), name='policy')
                elif isinstance(env.spec.action_space, Discrete):
                    if args.control == 'concurrent':
                        policies = [
                            CategoricalGRUPolicy(env_spec=env.spec, feature_network=feature_network,
                                                 hidden_dim=int(args.policy_hidden[0]),
                                                 name='policy_{}'.format(agid),
                                                 state_include_action=False if args.conv else True)
                            for agid in range(len(env.agents))
                        ]
                    policy = CategoricalGRUPolicy(env_spec=env.spec,
                                                  feature_network=feature_network,
                                                  hidden_dim=int(args.policy_hidden[0]),
                                                  name='policy', state_include_action=False if
                                                  args.conv else True)
                else:
                    raise NotImplementedError(env.spec.observation_space)

            elif args.recurrent == 'lstm':
                if isinstance(env.spec.action_space, Box):
                    if args.control == 'concurrent':
                        policies = [
                            GaussianLSTMPolicy(env_spec=env.spec, feature_network=feature_network,
                                               hidden_dim=int(args.policy_hidden),
                                               name='policy_{}'.format(agid))
                            for agid in range(len(env.agents))
                        ]
                    policy = GaussianLSTMPolicy(env_spec=env.spec, feature_network=feature_network,
                                                hidden_dim=int(args.policy_hidden), name='policy')
                elif isinstance(env.spec.action_space, Discrete):
                    if args.control == 'concurrent':
                        policies = [
                            CategoricalLSTMPolicy(env_spec=env.spec,
                                                  feature_network=feature_network,
                                                  hidden_dim=int(args.policy_hidden),
                                                  name='policy_{}'.format(agid))
                            for agid in range(len(env.agents))
                        ]
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
            assert isinstance(env.spec.action_space,
                              Discrete), "Only discrete action spaces support conv"
            feature_network = ConvNetwork(name='feature_net',
                                          input_shape=env.spec.observation_space.shape,
                                          output_dim=env.spec.action_space.n, conv_filters=chans,
                                          conv_filter_sizes=filts, conv_strides=strides,
                                          conv_pads=('VALID',) * len(chans),
                                          hidden_sizes=tuple(args.policy_hidden),
                                          hidden_nonlinearity=tf.nn.relu,
                                          output_nonlinearity=tf.nn.softmax)
            if args.control == 'concurrent':
                policies = [
                    CategoricalMLPPolicy(name='policy_{}'.format(agid), env_spec=env.spec,
                                         prob_network=feature_network)
                    for agid in range(len(env.agents))
                ]
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec,
                                          prob_network=feature_network)
        else:
            if isinstance(env.spec.action_space, Box):
                if args.control == 'concurrent':
                    policies = [
                        GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=tuple(args.policy_hidden),
                                          min_std=args.min_std, name='policy_{}'.format(agid))
                        for agid in range(len(env.agents))
                    ]
                policy = GaussianMLPPolicy(env_spec=env.spec,
                                           hidden_sizes=tuple(args.policy_hidden),
                                           min_std=args.min_std, name='policy')
            elif isinstance(env.spec.action_space, Discrete):
                if args.control == 'concurrent':
                    policies = [
                        CategoricalMLPPolicy(env_spec=env.spec,
                                             hidden_sizes=tuple(args.policy_hidden),
                                             name='policy_{}'.format(agid))
                        for agid in range(len(env.agents))
                    ]
                policy = CategoricalMLPPolicy(env_spec=env.spec,
                                              hidden_sizes=tuple(args.policy_hidden), name='policy')
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
                    policy = thGaussianGRUPolicy(
                        env_spec=env.spec,
                        feature_network=feature_network,
                        hidden_dim=int(args.policy_hidden[0]),)
                elif isinstance(env.spec.observation_space, thDiscrete):
                    policy = thCategoricalGRUPolicy(
                        env_spec=env.spec,
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
                policy = thDeterministicMLPPolicy(
                    env_spec=env.spec,
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
        self.env = env
        self.args = args
        # Parallel setup
        parallel_sampler.initialize(n_parallel=args.n_parallel)
        if args.seed is not None:
            set_seed(args.seed)
            parallel_sampler.set_seed(args.seed)

    def setup(self, env, policy, start_itr):

        if not self.args.algo == 'thddpg':
            # Baseline
            if self.args.baseline_type == 'linear':
                baseline = LinearFeatureBaseline(env_spec=env.spec)
            elif self.args.baseline_type == 'zero':
                baseline = ZeroBaseline(env_spec=env.spec)
            else:
                raise NotImplementedError(self.args.baseline_type)

            if self.args.control == 'concurrent':
                baseline = [baseline for _ in range(len(env.agents))]
        # Logger
        default_log_dir = config.LOG_DIR
        if self.args.log_dir is None:
            log_dir = osp.join(default_log_dir, self.args.exp_name)
        else:
            log_dir = self.args.log_dir

        tabular_log_file = osp.join(log_dir, self.args.tabular_log_file)
        text_log_file = osp.join(log_dir, self.args.text_log_file)
        params_log_file = osp.join(log_dir, self.args.params_log_file)

        logger.log_parameters_lite(params_log_file, self.args)
        logger.add_text_output(text_log_file)
        logger.add_tabular_output(tabular_log_file)
        prev_snapshot_dir = logger.get_snapshot_dir()
        prev_mode = logger.get_snapshot_mode()
        logger.set_snapshot_dir(log_dir)
        logger.set_snapshot_mode(self.args.snapshot_mode)
        logger.set_log_tabular_only(self.args.log_tabular_only)
        logger.push_prefix("[%s] " % self.args.exp_name)

        if self.args.algo == 'tftrpo':
            algo = MATRPO(env=env, policy_or_policies=policy, baseline_or_baselines=baseline,
                          batch_size=self.args.batch_size, start_itr=start_itr,
                          max_path_length=self.args.max_path_length, n_itr=self.args.n_iter,
                          discount=self.args.discount, gae_lambda=self.args.gae_lambda,
                          step_size=self.args.step_size, optimizer=ConjugateGradientOptimizer(
                              hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)) if
                          self.args.recurrent else None, ma_mode=self.args.control)
        elif self.args.algo == 'thddpg':
            qfunc = thContinuousMLPQFunction(env_spec=env.spec)
            if self.args.exp_strategy == 'ou':
                es = OUStrategy(env_spec=env.spec)
            elif self.args.exp_strategy == 'gauss':
                es = GaussianStrategy(env_spec=env.spec)
            else:
                raise NotImplementedError()

            algo = thDDPG(env=env, policy=policy, qf=qfunc, es=es, batch_size=self.args.batch_size,
                          max_path_length=self.args.max_path_length,
                          epoch_length=self.args.epoch_length,
                          min_pool_size=self.args.min_pool_size,
                          replay_pool_size=self.args.replay_pool_size, n_epochs=self.args.n_iter,
                          discount=self.args.discount, scale_reward=0.01,
                          qf_learning_rate=self.args.qfunc_lr,
                          policy_learning_rate=self.args.policy_lr,
                          eval_samples=self.args.eval_samples, mode=self.args.control)
        return algo

    def __call__(self, curriculum=None):
        # TODO Handle curriculum
        # Use curriculum config to come up with indexed set of environemnts
        # These envs will be of increasing difficulty
        # or rather define a distribution?
        # Initially the probability mass is on the simpler task
        # But as the agent gets better, the probability mass shifts to the more difficult task
        # Basically a dirichlet distribution?
        #
        # After some k training steps
        #   evaluate the current learned policy. on all the environments
        #   if the current policy performs "well" on an environment,
        #   decrease the probability of that environment (By shifting the dirichlet alpha values)
        #   and recalculate their probability
        #   otherwise retrain with the current
        # But when do we say, that we fail?
        env, policy = rllab_envpolicy_parser(self.env, self.args)
        if curriculum:
            import numpy as np
            from collections import defaultdict
            from rllab.misc.evaluate import evaluate
            task_dist = np.ones(len(curriculum.tasks))
            task_dist[0] = len(curriculum.tasks)
            min_reward = np.inf
            task_eval_reward = defaultdict(float)
            task_counts = defaultdict(int)
        if self.args.resume_from is not None:
            import joblib
            with tf.Session() as sess:
                data = joblib.load(self.args.resume_from)
                if 'algo' in data.keys():
                    algo = data['algo']
                    env = algo.env
                    policy = algo.policy_or_policies
                elif 'policy' in data.keys():
                    policy = data['policy']
                    env = data['env']
                    idx = data['itr']
        else:
            env, policy = rllab_envpolicy_parser(self.env, self.args)
            idx = 0
            algo = self.setup(env, policy, start_itr=idx)
            while True:
                for i in range(curriculum.n_trials):
                    logger.log("Running Curriculum trial: {}".format(i))
                    task_prob = np.random.dirichlet(task_dist)
                    task = np.random.choice(curriculum.tasks, p=task_prob)
                    env.set_param_values(task.prop)
                    setattr(algo, 'env', env)
                    setattr(algo, 'start_itr', idx)
                    setattr(algo, 'n_itr', idx + self.args.n_iter)
                    algo.train()
                    logger.log("Evaluating...")
                    with tf.Session() as sess:
                        sess.run(tf.initialize_all_variables())
                        ev = evaluate(env, policy, max_path_length=algo.max_path_length,
                                      n_paths=curriculum.n_trials, ma_mode=algo.ma_mode,
                                      disc=algo.discount)
                    task_eval_reward[task] += np.mean(ev['ret'])  # TODO
                    task_counts[task] += 1
                    idx += self.args.n_iter

                scores = []
                for i, task in enumerate(curriculum.tasks):
                    if task_counts[task] > 0:
                        score = 1.0 * task_eval_reward[task] / task_counts[task]
                        logger.log("task:{} {}".format(i, score))
                        scores.append(score)
                    else:
                        score.append(-np.inf)
                min_reward = min(min_reward, min(scores))
                rel_reward = scores[np.argmax(task_dist)]
                if rel_reward > curriculum.lesson_threshold:
                    logger.log("task: {} breached, reward: {}!".format(
                        np.argmax(task_dist), rel_reward))
                    task_dist = np.roll(task_dist, 1)
                if min_reward > curriculum.stop_threshold:
                    # Special SAVE?
                    break

        else:
            algo = self.setup(env, policy, start_itr=0)
            algo.train()
