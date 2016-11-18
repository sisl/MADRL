#!/usr/bin/env python
#
# File: run_waterworld.py
#
# Created: Wednesday, August 31 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser
from runners.curriculum import Curriculum

from madrl_environments.pursuit import MAWaterWorld
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('radius', float, 0.015, 'Radius of agents'),
    ('n_evaders', int, 10, ''),
    ('n_pursuers', int, 8, ''),
    ('n_poison', int, 10, ''),
    ('n_coop', int, 4, ''),
    ('n_sensors', int, 30, ''),
    ('sensor_range', int, 0.2, ''),
    ('food_reward', float, 10, ''),
    ('poison_reward', float, -1, ''),
    ('encounter_reward', float, 0.01, ''),
    ('reward_mech', str, 'local', ''),
    ('noid', str, None, ''),
    ('speed_features', int, 1, ''),
    ('buffer_size', int, 1, ''),
    ('curriculum', str, None, ''),
]
# yapf: enable


def main(parser):
    mode = parser._mode
    args = parser.args
    env = MAWaterWorld(args.n_pursuers, args.n_evaders, args.n_coop, args.n_poison,
                       radius=args.radius, n_sensors=args.n_sensors, food_reward=args.food_reward,
                       poison_reward=args.poison_reward, encounter_reward=args.encounter_reward,
                       reward_mech=args.reward_mech, sensor_range=args.sensor_range,
                       obstacle_loc=None, addid=True if not args.noid else False,
                       speed_features=bool(args.speed_features))

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        from runners.rurltools import RLToolsRunner
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    if args.curriculum:
        curr = Curriculum(args.curriculum)
        run(curr)
    else:
        run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
