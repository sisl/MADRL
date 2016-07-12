import numpy as np

import optim
import tfutil
import util
from policy import StochasticPolicy
from sampler import SimpleSampler


class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError()


class SamplingPolicyOptimizer(RLAlgorithm):
    def __init__(self, env, policy, baseline, step_func,
                 discount=0.99, gae_lambda=1, n_iter=500, start_iter=0,
                 center_adv=True, positive_adv=False,
                 store_paths=False, whole_paths=True,
                 max_traj_len=200, batch_size=32,
                 **kwargs):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.step_func = step_func
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        # TODO obsfeat
        self.sampler = SimpleSampler(self, max_traj_len, batch_size)
        self.total_time = 0.0

    def train(self, sess, log):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % 20 == 0:
                log.write_snapshot(sess, self.policy, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                if itr == 0:
                    # extra batch to init std
                    trajbatch0, _ = self.sampler.sample(sess, itr)
                    self.policy.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                    self.baseline.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Compute baseline
            with util.Timer() as t_base:
                trajbatch_vals, base_info_fields = self.sampler.process(sess, itr, trajbatch)

            # Take the policy grad step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params(sess)
                step_print_fields = self.step_func(sess, self.policy, params0_P, trajbatch, trajbatch_vals['advantage'])
                self.policy.update_obsnorm(sess, trajbatch.obsfeat.stacked)

        # LOG
        self.total_time += t_all.dt

        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + [
            ('vf_r2', trajbatch_vals['v_r'], float),
            ('tdv_r2', trajbatch_vals['tv_r'], float),
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float), # entropy of action distribution
            ('dx', util.maxnorm(params0_P - self.policy.get_params(sess)), float) # max parameter different from last iteration
        ] + base_info_fields + step_print_fields + [
            ('tsamp', t_sample.dt, float), # Time for sampling
            ('tbase', t_base.dt, float),   # Time for advantage/baseline computation
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields


def TRPO(max_kl, subsample_hvp_frac=.25, damping=1e-2, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):

    def trpo_step(sess, policy, params0_P, trajbatch, advantages):
        # standardize advantage
        advstacked_N = util.standardized(advantages.stacked)

        # Compute objective, KL divergence and gradietns at init point
        feed = (trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked, advstacked_N)

        step_info = policy._ngstep(sess, feed, max_kl=max_kl, damping=damping, subsample_hvp_frac=subsample_hvp_frac, grad_stop_tol=grad_stop_tol)
        return step_info
    return trpo_step
