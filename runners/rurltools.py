from __future__ import absolute_import, print_function

import json

import numpy as np
import tensorflow as tf
from gym import spaces

from rltools import log, util
from rltools.algos.policyopt import TRPO, SamplingPolicyOptimizer, ConcurrentPolicyOptimizer
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.categorical import CategoricalMLPPolicy
from rltools.policy.gaussian import GaussianGRUPolicy, GaussianMLPPolicy
from rltools.samplers.parallel import ParallelSampler
from rltools.samplers.serial import DecSampler, SimpleSampler, ConcSampler

from runners import tonamedtuple


def rltools_envpolicy_parser(env, args):
    if isinstance(args, dict):
        args = tonamedtuple(args)

        # XXX
        # Should be handled in the environment?
        # shape mucking is incorrect for image envs
    if args.control == 'centralized':
        obs_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                               high=env.agents[0].observation_space.high[0],
                               shape=(env.agents[0].observation_space.shape[0] * len(env.agents),))
        action_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                                  high=env.agents[0].observation_space.high[0],
                                  shape=(env.agents[0].action_space.shape[0] * len(env.agents),))
    else:
        obs_space = env.agents[0].observation_space
        action_space = env.agents[0].action_space

    if args.recurrent:
        if args.recurrent == 'gru':
            if isinstance(action_space, spaces.Box):
                if args.control == 'concurrent':
                    policies = [GaussianGRUPolicy(env.agents[agid].observation_space,
                                                  env.agents[agid].action_space,
                                                  hidden_spec=args.policy_hidden_spec,
                                                  min_stdev=args.min_std, init_logstdev=0.,
                                                  enable_obsnorm=args.enable_obsnorm,
                                                  state_include_action=False,
                                                  varscope_name='policy_{}'.format(agid))
                                for agid in range(len(env.agents))]

                policy = GaussianGRUPolicy(obs_space, action_space,
                                           hidden_spec=args.policy_hidden_spec,
                                           min_stdev=args.min_std, init_logstdev=0.,
                                           enable_obsnorm=args.enable_obsnorm,
                                           state_include_action=False, varscope_name='policy')

            elif isinstance(action_space, spaces.Discrete):
                raise NotImplementedError(args.recurrent)
        else:
            raise NotImplementedError()
    else:
        if isinstance(action_space, spaces.Box):
            if args.control == 'concurrent':
                policies = [GaussianMLPPolicy(env.agents[agid].observation_space,
                                              env.agents[agid].action_space,
                                              hidden_spec=args.policy_hidden_spec,
                                              min_stdev=args.min_std, init_logstdev=0.,
                                              enable_obsnorm=args.enable_obsnorm,
                                              varscope_name='{}_policy'.format(agid))
                            for agid in range(len(env.agents))]

            policy = GaussianMLPPolicy(obs_space, action_space, hidden_spec=args.policy_hidden_spec,
                                       min_stdev=args.min_std, init_logstdev=0.,
                                       enable_obsnorm=args.enable_obsnorm, varscope_name='policy')

        elif isinstance(action_space, spaces.Discrete):
            if args.control == 'concurrent':
                policies = [CategoricalMLPPolicy(env.agents[agid].observation_space,
                                                 env.agents[agid].action_space,
                                                 hidden_spec=args.policy_hidden_spec,
                                                 enable_obsnorm=args.enable_obsnorm,
                                                 varscope_name='policy_{}'.format(agid))
                            for agid in range(len(env.agents))]

            policy = CategoricalMLPPolicy(obs_space, action_space,
                                          hidden_spec=args.policy_hidden_spec,
                                          enable_obsnorm=args.enable_obsnorm,
                                          varscope_name='policy')

        else:
            raise NotImplementedError()

    if args.control == 'concurrent':
        return env, policies, policy
    else:
        return env, None, policy


class RLToolsRunner(object):

    def __init__(self, env, args):
        self.args = args
        env, policies, policy = rltools_envpolicy_parser(env, args)
        if args.baseline_type == 'linear':
            if args.control == 'concurrent':
                baselines = [LinearFeatureBaseline(env.agents[agid].observation_space,
                                                   enable_obsnorm=args.enable_obsnorm,
                                                   varscope_name='baseline_{}'.format(agid))
                             for agid in range(len(env.agents))]
            else:
                baseline = LinearFeatureBaseline(policy.observation_space,
                                                 enable_obsnorm=args.enable_obsnorm,
                                                 varscope_name='baseline')

        elif args.baseline_type == 'mlp':
            if args.control == 'concurrent':
                baselines = [MLPBaseline(env.agents[agid].observation_space,
                                         hidden_spec=args.baseline_hidden_spec,
                                         enable_obsnorm=args.enable_obsnorm,
                                         enable_vnorm=args.enable_vnorm, max_kl=args.vf_max_kl,
                                         damping=args.vf_cg_damping,
                                         time_scale=1. / args.max_traj_len,
                                         varscope_name='{}_baseline'.format(agid))
                             for agid in range(len(env.agents))]
            else:
                baseline = MLPBaseline(policy.observation_space,
                                       hidden_spec=args.baseline_hidden_spec,
                                       enable_obsnorm=args.enable_obsnorm,
                                       enable_vnorm=args.enable_vnorm, max_kl=args.vf_max_kl,
                                       damping=args.vf_cg_damping,
                                       time_scale=1. / args.max_traj_len, varscope_name='baseline')

        elif args.baseline_type == 'zero':
            if args.control == 'concurrent':
                baselines = [ZeroBaseline(env.agents[agid].observation_space)
                             for agid in range(len(env.agents))]
            else:
                baseline = ZeroBaseline(policy.observation_space)
        else:
            raise NotImplementedError()

        if args.sampler == 'simple':
            if args.control == 'centralized':
                sampler_cls = SimpleSampler
            elif args.control == 'decentralized':
                sampler_cls = DecSampler
            elif args.control == 'concurrent':
                sampler_cls = ConcSampler
            else:
                raise NotImplementedError()
            sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                                n_timesteps_min=args.n_timesteps_min,
                                n_timesteps_max=args.n_timesteps_max,
                                timestep_rate=args.timestep_rate, adaptive=args.adaptive_batch,
                                enable_rewnorm=args.enable_rewnorm)
        elif args.sampler == 'parallel':
            sampler_cls = ParallelSampler
            sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                                n_timesteps_min=args.n_timesteps_min,
                                n_timesteps_max=args.n_timesteps_max,
                                timestep_rate=args.timestep_rate, adaptive=args.adaptive_batch,
                                enable_rewnorm=args.enable_rewnorm, n_workers=args.sampler_workers,
                                mode=args.control, discard_extra=False)

        else:
            raise NotImplementedError()

        step_func = TRPO(max_kl=args.max_kl)
        if args.control == 'concurrent':
            self.algo = ConcurrentPolicyOptimizer(env=env, policies=policies, baselines=baselines,
                                                  step_func=step_func, discount=args.discount,
                                                  gae_lambda=args.gae_lambda,
                                                  sampler_cls=sampler_cls,
                                                  sampler_args=sampler_args, n_iter=args.n_iter,
                                                  target_policy=policy,
                                                  interp_alpha=args.interp_alpha)
        else:
            self.algo = SamplingPolicyOptimizer(env=env, policy=policy, baseline=baseline,
                                                step_func=step_func, discount=args.discount,
                                                gae_lambda=args.gae_lambda, sampler_cls=sampler_cls,
                                                sampler_args=sampler_args, n_iter=args.n_iter)

        argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
        util.header(argstr)
        self.log_f = log.TrainingLog(args.log, [('args', argstr)], debug=args.debug)

    def __call__(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(self.args.tblog, graph=sess.graph)
            self.algo.train(sess, self.log_f, self.args.save_freq, blend_freq=self.args.blend_freq,
                            keep_kmax=self.args.keep_kmax,
                            blend_eval_trajs=self.args.blend_eval_trajs)
