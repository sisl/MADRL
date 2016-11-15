from madrl_environments import StandardizedEnv
from madrl_environments.pursuit import MAWaterWorld
from rllabwrapper import RLLabEnv
from rllab.sampler import parallel_sampler
from sandbox.rocky.tf.algos.ma_trpo import MATRPO
from sandbox.rocky.tf.envs.base import MATfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

parallel_sampler.initialize(n_parallel=2)
env = StandardizedEnv(MAWaterWorld(3, 10, 2, 5))
env = MATfEnv(RLLabEnv(env, ma_mode='decentralized'))

policy = GaussianGRUPolicy(env_spec=env.spec, name='policy')

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = MATRPO(env=env, policy_or_policies=policy, baseline_or_baselines=baseline, batch_size=8000,
              max_path_length=200, n_itr=500, discount=0.99, step_size=0.01,
              optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
              ma_mode='decentralized')
# policies = [GaussianGRUPolicy(env_spec=env.spec, name='policy_{}'.format(i)) for i in range(3)]
# baselines = [LinearFeatureBaseline(env_spec=env.spec) for _ in range(3)]
# algo = MATRPO(env=env, policy_or_policies=policies, baseline_or_baselines=baselines,
#               batch_size=8000, max_path_length=200, n_itr=500, discount=0.99, step_size=0.01,
#               optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
#               ma_mode='concurrent')

algo.train()
