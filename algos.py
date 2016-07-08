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
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Compute baseline
            with util.Timer() as t_base:
                trajbatch_vals, base_info_fields = self.sampler.process(itr, trajbatch)

            # Take the policy grad step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params(sess)
                step_print_fields = self.step_func(sess, self.policy, params0_P, trajbatch, trajbatch_vals['advantage'])


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


def TRPO(max_kl, subsample_hvp_frac=.1, damping=1e-2, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):

    def trpo_step(sess, policy, params0_P, trajbatch, advantages):
        # standardize advantage
        advstacked_N = util.standardized(advantages.stacked)

        # Compute objective, KL divergence and gradietns at init point
        feed = (trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked, advstacked_N)
        reinfobj0, kl0, reinfobjgrad0 = policy.compute_reinfobj_kl_with_grad(sess, *feed)
        # info0 = policy.compute(sess, feed, reinfobj=True, kl=True, reinfobjgrad=True, klgrad=True)
        gnorm = util.maxnorm(reinfobjgrad0)
        assert np.allclose(kl0, 0.0, atol=1e-06), "Initial KL divergence is %.7f, but should be 0" % (kl0)
        # if np.allclose(info0.reinfobjgrad_P, 0):
        # Terminate early if gradients are too small
        if gnorm < grad_stop_tol:
            reinfobj1 = reinfobj0
            kl1 = kl0
            reinfobjgrad1 = reinfobjgrad0
            num_bt_steps = 0
        else:
            # Take constrained ng step

            # Data subsample for Hessain vector products
            subsamp_feed = feed if subsample_hvp_frac is None else tfutil.subsample_feed(feed, subsample_hvp_frac)

            def hvp_klgrad_func(p):
                with policy.try_params(sess, p):
                    return policy.compute_klgrad(sess, subsamp_feed[0], subsamp_feed[2])[0]

            # Line search objective
            def obj_and_kl_func(p):
                with policy.try_params(sess, p):
                    reinfobj, kl = policy.compute_reinfobj_kl(sess, *feed)
                return -reinfobj, kl

            params1_P, num_bt_steps = optim.ngstep(
                x0=params0_P,    # current
                obj0=-reinfobj0, # current
                objgrad0=-reinfobjgrad0,
                obj_and_kl_func=obj_and_kl_func,
                hvpx0_func=hvp_klgrad_func,
                max_kl=max_kl,
                damping=damping,
                max_cg_iter=max_cg_iter,
                enable_bt=enable_bt
            )
            policy.set_params(sess, params1_P)
            reinfobj1, kl1 = policy.compute_reinfobj_kl(sess, *feed)

        return [
            ('dl', reinfobj1 - reinfobj0, float), # improvement of objective
            ('kl', kl1, float),                        # kl cost of solution
            ('bt', num_bt_steps, int)                     # number of backtracking steps
        ]
    return trpo_step
