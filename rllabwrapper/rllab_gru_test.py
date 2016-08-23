from madrl_environments import StandardizedEnv
from madrl_environments.pursuit import MAWaterWorld
from rllabwrapper import RLLabEnv

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

env = StandardizedEnv(MAWaterWorld(3, 10, 2, 5))
env = RLLabEnv(env)

policy = GaussianGRUPolicy(env_spec=env.spec, hidden_sizes=(32,))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(env=env,
            policy=policy,
            baseline=baseline,
            batch_size=8000,
            max_path_length=200,
            n_itr=500,
            discount=0.99,
            step_size=0.01,
            mode='decentralized',)

algo.train()
