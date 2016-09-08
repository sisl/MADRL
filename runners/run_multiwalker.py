#!/usr/bin/env python
#
# File: run_multiwalker.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser

from madrl_environments.walker.multi_walker import MultiWalkerEnv
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_walkers', int, 2, ''),
    ('position_noise', float, 1e-3, ''),
    ('angle_noise', float, 1e-3, ''),
    ('reward_mech', str, 'local', ''),
    ('forward_reward', float, 1.0, ''),
    ('fall_reward', float, -100.0, ''),
    ('drop_reward', float, -100.0, ''),
    ('terminate_on_fall', int, 1, ''),
    ('buffer_size', int, 1, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args

    env = MultiWalkerEnv(n_walkers=args.n_walkers, position_noise=args.position_noise,
                         angle_noise=args.angle_noise, reward_mech=args.reward_mech,
                         forward_reward=args.forward_reward, fall_reward=args.fall_reward,
                         drop_reward=args.drop_reward,
                         terminate_on_fall=bool(args.terminate_on_fall))

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
