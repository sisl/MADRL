#!/usr/bin/env python
#
# File: vis_waterworld.py
#
# Created: Thursday, July 14 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import pprint
import os
import os.path

from gym import spaces
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util
import rltools.samplers
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import MAWaterWorld
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy

from vis import Evaluator, Visualizer, FileHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--heuristic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--n_trajs', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--same_con_pol', action='store_true')
    args = parser.parse_args()

    fh = FileHandler(args.filename)

    env = MAWaterWorld(fh.train_args['n_pursuers'],
                       fh.train_args['n_evaders'],
                       fh.train_args['n_coop'],
                       fh.train_args['n_poison'],
                       n_sensors=fh.train_args['n_sensors'],
                       food_reward=fh.train_args['food_reward'],
                       poison_reward=fh.train_args['poison_reward'],
                       encounter_reward=fh.train_args['encounter_reward'],)

    if fh.train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, fh.train_args['buffer_size'])

    if args.heuristic:
        from heuristic.waterworld import WaterworldHeuristicPolicy
        hpolicy = WaterworldHeuristicPolicy(env.agents[0].observation_space,
                                            env.agents[0].action_space)

    if args.evaluate:
        minion = Evaluator(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                           'heuristic' if args.heuristic else fh.mode)
        evr = minion(fh.filename, file_key=fh.file_key, same_con_pol=args.same_con_pol,
                     hpolicy=hpolicy)
        from tabulate import tabulate
        print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                            fh.mode)
        rew, info = minion(fh.filename, file_key=fh.file_key, vid=args.vid)
        pprint.pprint(rew)
        pprint.pprint(info)


if __name__ == '__main__':
    main()
