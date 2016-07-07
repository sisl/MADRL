import numpy as np

from sampler import SimpleSampler
from policy import Policy
import util
import optim

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

        self.sampler = SimpleSampler(self, max_traj_len, batch_size)
        self.total_time = 0.0

    def train(self, sess, logfilename):
        for itr in range(self.start_iter, self.n_iter):
            print_fields = self.step(sess, itr)


    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Compute baseline
            with util.Timer() as t_base:
                trajbatch_vals = self.sampler.process(sess, itr, trajbatch)

            # Take the policy grad step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params(sess)
                step_print_fields = self.step_func(sess, self.policy, params0_P, trajbatch, trajbatch_vals['advantage'])


        # LOG
        self.total_time += t_all.dt
        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + [
            # entropy of action distribution
            # max parameter different from last iteration
        ] + step_print_fields + [
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
        feed = Policy.Feed(trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked, advstacked_N, kl_cost_coeff=None)
        info0 = policy.compute(sess, feed, reinfobj=True, kl=True, reinfobjgrad=True, klgrad=True)
        gnorm = util.maxnorm(info0.reinfobjgrad_P)
        assert np.allclose(info0.kl, 0), "Initial KL divergence is %.7f, but should be 0" % (kl0)
        # if np.allclose(info0.reinfobjgrad_P, 0):
        # Terminate early if gradients are too small
        if gnorm < grad_stop_tol:
            #
            info1 = info0
            num_bt_steps = 0
        else:
            # Take constrained ng step

            # Data subsample for Hessain vector products
            subsamp_feed = feed if subsample_hvp_frac is None else Policy.subsample_feed(feed, int(subsample_hvp_frac*feed.obsfeat_B_Fd.shape[0]))

            def hvp_klgrad_func(p):
                with policy.try_params(sess, p):
                    return policy.compute(sess, subsamp_feed, klgrad=True).klgrad_P

            # Line search objective
            def obj_and_kl_func(p):
                with policy.try_params(sess, p):
                    info = policy.compute(sess, feed, reinfobj=True, kl=True)
                return -info.reinfobj, info.kl

            params1_P, num_bt_steps = optim.ngstep(
                x0=params0_P,    # current
                obj0=-info0.reinfobj, # current
                objgrad0=-info0.reinfobjgrad_P,
                obj_and_kl_func=obj_and_kl_func,
                hvp_klgrad_func=hvp_klgrad_func,
                max_kl=max_kl,
                klgrad0=info0.klgrad_P,
                damping=damping,
                max_cg_iter=max_cg_iter,
                enable_bt=enable_bt
            )
            policy.set_params(sess, params1_P)
            info1 = policy.compute(sess, feed, reinfobj=True, kl=True)

        return [
            ('dl', info1.reinfobj - info0.reinfobj, float), # improvement of objective
            ('kl', info1.kl, float),                        # kl cost of solution
            ('bt', num_bt_steps, int)                     # number of backtracking steps
        ]
    return trpo_step
