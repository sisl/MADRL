#!/usr/bin/env python
#
# File: run_hostage.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser, comma_sep_ints
from runners.rurllab import RLLabRunner
from runners.rurltools import RLToolsRunner
from madrl_environments.hostage import ContinuousHostage
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_good', int, 4, ''),
    ('n_hostages', int, 12, ''),
    ('n_bad', int, 12, ''),
    ('n_coop_save', int, 4, ''),
    ('n_coop_avoid', int, 1, ''),
    ('radius', float, 0.015, ''),
    ('bomb_radius', float, 0.03, ''),
    ('key_loc', comma_sep_ints, None, ''),
    ('bad_speed', float, 0.01, ''),
    ('n_sensors', int, 30, ''),
    ('sensor_range', float, 0.2, ''),
    ('save_reward', float, 10, ''),
    ('hit_reward', float, -1, ''),
    ('encounter_reward', float, 0.01, ''),
    ('bomb_reward', float, -20, ''),
    ('control_penalty', float, -0.1, ''),
    ('reward_mech', str, 'local', ''),
    ('buffer_size', int, 1, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args

    env = ContinuousHostage(n_good=args.n_good,
                            n_hostages=args.n_hostages,
                            n_bad=args.n_bad,
                            n_coop_save=args.n_coop_save,
                            n_coop_avoid=args.n_coop_avoid,
                            radius=args.radius,
                            key_loc=args.key_loc,
                            bad_speed=args.bad_speed,
                            n_sensors=args.n_sensors,
                            sensor_range=args.sensor_range,
                            save_reward=args.save_reward,
                            hit_reward=args.hit_reward,
                            encounter_reward=args.encounter_reward,
                            bomb_reward=args.bomb_reward,
                            bomb_radius=args.bomb_radius,
                            control_penalty=args.control_penalty,
                            reward_mech=args.reward_mech,)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
