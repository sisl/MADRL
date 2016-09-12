import numpy as np

from rltools.policy import Policy

STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
SPEED = 0.29  # Will fall forward on higher speed
SUPPORT_KNEE_ANGLE = +0.1


class MultiWalkerHeuristicPolicy(Policy):

    def __init__(self, observation_space, action_space):
        super(MultiWalkerHeuristicPolicy, self).__init__(observation_space, action_space)

    def sample_actions(self, obs_B_Do, deterministic=True):

        n_agents = obs_B_Do.shape[0]
        actions = np.zeros((n_agents, 4))

        for i in xrange(n_agents):
            a = np.zeros(4)
            s = obs_B_Do[i]
            state = STAY_ON_ONE_LEG
            moving_leg = 0
            supporting_leg = 1 - moving_leg
            supporting_knee_angle = SUPPORT_KNEE_ANGLE

            contact0 = s[8]
            contact1 = s[13]
            moving_s_base = 4 + 5 * moving_leg
            supporting_s_base = 4 + 5 * supporting_leg

            hip_targ = [None, None]  # -0.8 .. +1.1
            knee_targ = [None, None]  # -0.6 .. +0.9
            hip_todo = [0.0, 0.0]
            knee_todo = [0.0, 0.0]

            if state == STAY_ON_ONE_LEG:
                hip_targ[moving_leg] = 1.1
                knee_targ[moving_leg] = -0.6
                supporting_knee_angle += 0.03
                if s[2] > SPEED:
                    supporting_knee_angle += 0.03
                supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                    state = PUT_OTHER_DOWN
            if state == PUT_OTHER_DOWN:
                hip_targ[moving_leg] = +0.1
                knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                knee_targ[supporting_leg] = supporting_knee_angle
                if s[moving_s_base + 4]:
                    state = PUSH_OFF
                    supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
            if state == PUSH_OFF:
                knee_targ[moving_leg] = supporting_knee_angle
                knee_targ[supporting_leg] = +1.0
                if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                    state = STAY_ON_ONE_LEG
                    moving_leg = 1 - moving_leg
                    supporting_leg = 1 - moving_leg

            if hip_targ[0]:
                hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
            if hip_targ[1]:
                hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
            if knee_targ[0]:
                knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
            if knee_targ[1]:
                knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

            hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
            hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
            knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
            knee_todo[1] -= 15.0 * s[3]

            a[0] = hip_todo[0]
            a[1] = knee_todo[0]
            a[2] = hip_todo[1]
            a[3] = knee_todo[1]
            a = np.clip(0.5 * a, -1.0, 1.0)

            actions[i, :] = a

        fake_actiondist = np.concatenate([np.zeros((n_agents, 4)), np.ones((n_agents, 4))])
        return actions, fake_actiondist


if __name__ == '__main__':
    from madrl_environments.walker.multi_walker import MultiWalkerEnv
    from vis import Visualizer
    import pprint
    env = MultiWalkerEnv(n_walkers=3)
    train_args = {'discount': 0.99, 'control': 'decentralized'}
    vis = Visualizer(env, train_args, 500, 1, True, 'heuristic')

    rew, info = vis(None, hpolicy=MultiWalkerHeuristicPolicy(env.agents[0].observation_space,
                                                             env.agents[0].observation_space))

    pprint.pprint(rew)
    pprint.pprint(info)
