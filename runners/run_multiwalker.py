#!/usr/bin/env python
#
# File: run_multiwalker.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser
from runners.rurllab import RLLabRunner
from runners.rurltools import RLToolsRunner
from madrl_environments.walker.multi_walker import MultiWalkerEnv
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_walkers', int, 2, ''),
    ('position_noise', float, 1e-3, ''),
    ('angle_noise', float, 1e-3, ''),
    ('buffer_size', int, 1, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args

    env = MultiWalkerEnv(n_walkers=args.n_walkers, position_noise=args.position_noise,
                         angle_noise=args.angle_noise)

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
