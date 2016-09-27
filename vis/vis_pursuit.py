#!/usr/bin/env python
#
# File: vis_pursuit.py
#
# Created: Wednesday, September 14 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import pprint
import os
import os.path
import numpy as np

from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from madrl_environments import StandardizedEnv, ObservationBuffer

from vis import Evaluator, Visualizer, FileHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--heuristic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--n_trajs', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--same_con_pol', action='store_true')
    args = parser.parse_args()

    fh = FileHandler(args.filename)

    map_pool = np.load(
        os.path.join('/scratch/megorov/deeprl/MADRL/runners/maps/', os.path.basename(fh.train_args[
            'map_file'])))
    env = PursuitEvade(map_pool, n_evaders=fh.train_args['n_evaders'],
                       n_pursuers=fh.train_args['n_pursuers'], obs_range=fh.train_args['obs_range'],
                       n_catch=fh.train_args['n_catch'], urgency_reward=fh.train_args['urgency'],
                       surround=bool(fh.train_args['surround']),
                       sample_maps=bool(fh.train_args['sample_maps']),
                       flatten=bool(fh.train_args['flatten']), reward_mech='global',
                       catchr=fh.train_args['catchr'], term_pursuit=fh.train_args['term_pursuit'])

    if fh.train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, fh.train_args['buffer_size'])

    hpolicy = None
    if args.heuristic:
        from heuristics.pursuit import PursuitHeuristicPolicy
        hpolicy = PursuitHeuristicPolicy(env.agents[0].observation_space,
                                         env.agents[0].action_space)

    if args.evaluate:
        minion = Evaluator(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                           'heuristic' if args.heuristic else fh.mode)
        evr = minion(fh.filename, file_key=fh.file_key, same_con_pol=args.same_con_pol,
                     hpolicy=hpolicy)
        from tabulate import tabulate
        print(evr)
        print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                            fh.mode)
        rew, info = minion(fh.filename, file_key=fh.file_key, vid=args.vid)
        pprint.pprint(rew)
        pprint.pprint(info)


if __name__ == '__main__':
    main()
