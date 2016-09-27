import numpy as np

from rltools.policy import Policy


class WaterworldHeuristicPolicy(Policy):

    def __init__(self, observation_space, action_space):
        super(WaterworldHeuristicPolicy, self).__init__(observation_space, action_space)

    def sample_actions(self, obs_B_Do, deterministic=True):
        # Obs space
        # 0:K    -> obdist
        # K:2K   -> evdist
        # 2K:3K  -> evspeed
        # 3K:4K  -> podist
        # 4K:5K  -> pospeed
        # 5K:6K  -> pudist (allies)
        # 6K:7K  -> puspeed (allies)
        # 7K     -> colliding with ev
        # 7K+1   -> colliding with po
        # [7K+2] -> id

        # Act space
        # 0 -> x, 1-> y acceleration
        B = obs_B_Do.shape[0]
        K = obs_B_Do.shape[1] // 7
        angles_K = np.linspace(0., 2. * np.pi, K + 1)[:-1]

        vecs_K = np.c_[np.cos(angles_K), np.sin(angles_K)]

        obs_avoidance_action_B_2 = -np.sum(obs_B_Do[:, 0:K][..., None] * np.expand_dims(vecs_K, 0),
                                           axis=1)

        ev_catch_action_B_2 = np.sum(obs_B_Do[:, K:2 * K][..., None] * np.expand_dims(vecs_K, 0),
                                     axis=1)

        po_avoidance_action_B_2 = -np.sum(obs_B_Do[:, 3 * K:4 * K][..., None] *
                                          np.expand_dims(vecs_K, 0), axis=1)

        pu_closer_action_B_2 = np.sum(obs_B_Do[:, 5 * K:6 * K][..., None] *
                                      np.expand_dims(vecs_K, 0), axis=1) / 2

        ev_catch_action_B_2[obs_B_Do[:, 7 * K] > 0] *= 1.5
        po_avoidance_action_B_2[obs_B_Do[:, 7 * K + 1] > 0] *= 1.5

        actions_B_2 = obs_avoidance_action_B_2 + ev_catch_action_B_2 + po_avoidance_action_B_2 + pu_closer_action_B_2
        norm = np.linalg.norm(actions_B_2)
        if norm > 0:
            actions_B_2 /= norm
        else:
            actions_B_2 = np.zeros((B, 2))

        # actions_B_2 = np.random.randn(B, 2)
        fake_actiondist = np.concatenate([np.zeros((B, 2)), np.ones((B, 2))])
        return actions_B_2, fake_actiondist

    def get_state(self):
        return []

    def set_state(self, *args):
        pass


if __name__ == '__main__':
    from madrl_environments.pursuit import MAWaterWorld
    from vis import Visualizer
    import pprint
    env = MAWaterWorld(n_evaders=10, n_pursuers=8, n_poison=10, n_coop=4, n_sensors=30,
                       food_reward=10, poison_reward=-1, encounter_reward=0.01)
    train_args = {'discount': 0.99, 'control': 'decentralized'}

    vis = Visualizer(env, train_args, 500, 1, True, 'heuristic')

    rew, info = vis(None, hpolicy=WaterworldHeuristicPolicy(env.agents[0].observation_space,
                                                            env.agents[0].action_space))
    pprint.pprint(rew)
    pprint.pprint(info)
