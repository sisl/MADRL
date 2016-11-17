#!/usr/bin/env python
#
# File: run_multiwalker.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import numpy as np
from runners import RunnerParser

from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_evaders', int, 2, ''),
    ('n_pursuers', int, 2, ''),
    ('obs_range', int, 3, ''),
    ('map_size', str, '10,10', ''),
    ('map_type', str, 'rectangle', ''),
    ('n_catch', int, 2, ''),
    ('urgency', float, 0.0, ''),
    ('surround', int, 1, ''),
    ('map_file', str, None, ''),
    ('sample_maps', int, 0, ''),
    ('flatten', int, 1, ''),
    ('reward_mech', str, 'local', ''),
    ('catchr', float, 0.1, ''),
    ('term_pursuit', float, 5.0, ''),
    ('buffer_size', int, 1, ''),
    ('noid', str, None, ''),
]
# yapf: enable


def main(parser):
    mode = parser._mode
    args = parser.args

    if args.map_file:
        map_pool = np.load(args.map_file)
    else:
        if args.map_type == 'rectangle':
            env_map = TwoDMaps.rectangle_map(*map(int, args.map_size.split(',')))
        elif args.map_type == 'complex':
            env_map = TwoDMaps.complex_map(*map(int, args.map_size.split(',')))
        else:
            raise NotImplementedError()
        map_pool = [env_map]

    env = PursuitEvade(map_pool, n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
                       obs_range=args.obs_range, n_catch=args.n_catch, urgency_reward=args.urgency,
                       surround=bool(args.surround), sample_maps=bool(args.sample_maps),
                       flatten=bool(args.flatten), reward_mech=args.reward_mech, catchr=args.catchr,
                       term_pursuit=args.term_pursuit, include_id=not bool(args.noid))

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

    run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
