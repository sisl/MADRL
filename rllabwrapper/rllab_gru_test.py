from madrl_environments.walker.multi_walker import MultiWalkerEnv
from rllab_wrapper import GymEnv

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy


n_walkers = 1

env = MultiWalkerEnv(n_walkers=n_walkers)
env = normalize(GymEnv(env))

#policy = GaussianMLPPolicy(
#    env_spec=env.spec,
#    hidden_sizes=(128, 128)
#)
policy = GaussianGRUPolicy(
    env_spec=env.spec,
    hidden_sizes=(64,)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=8000,
    max_path_length=200,
    n_itr=500,
    discount=0.99,
    step_size=0.01,
)

algo.train()
