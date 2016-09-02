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

from vis import Evaluator, Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--n_trajs', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    args = parser.parse_args()

    # Handle remote files
    if ':' in args.filename:
        import uuid

        tmpfilename = str(uuid.uuid4())
        if 'h5' in args.filename.split('.')[-1]:
            os.system('rsync -avrz {}.h5 /tmp/{}.h5'.format(
                args.filename.split('.')[0], tmpfilename))
            newfilename = '/tmp/{}.{}'.format(tmpfilename, args.filename.split('.')[-1])
            args.filename = newfilename
        else:
            os.system('rsync -avrz {} /tmp/{}.pkl'.format(args.filename, tmpfilename))
            newfilename = '/tmp/{}.pkl'.format(tmpfilename)
            args.filename = newfilename
            # json file?
            # TODO

        # Load file
    if 'h5' in args.filename.split('.')[-1]:
        mode = 'rltools'
        filename, file_key = rltools.util.split_h5_name(args.filename)
        print('Loading parameters from {} in {}'.format(file_key, filename))
        with h5py.File(filename, 'r') as f:
            train_args = json.loads(f.attrs['args'])
            dset = f[file_key]
            pprint.pprint(dict(dset.attrs))
    else:
        # Pickle file
        mode = 'rllab'
        policy_dir = os.path.dirname(args.filename)
        params_file = os.path.join(policy_dir, 'params.json')
        filename = args.filename
        file_key = None
        print('Loading parameters from {} in {}'.format('params.json', policy_dir))
        with open(params_file, 'r') as df:
            train_args = json.load(df)

    env = MAWaterWorld(train_args['n_pursuers'],
                       train_args['n_evaders'],
                       train_args['n_coop'],
                       train_args['n_poison'],
                       n_sensors=train_args['n_sensors'],
                       food_reward=train_args['food_reward'],
                       poison_reward=train_args['poison_reward'],
                       encounter_reward=train_args['encounter_reward'],)

    if train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, train_args['buffer_size'])

    if args.evaluate:
        minion = Evaluator(env, train_args, args.n_steps, args.n_trajs, args.deterministic, mode)
        evr = minion(filename, file_key=file_key)
        from tabulate import tabulate
        print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, train_args, args.n_steps, args.n_trajs, args.deterministic, mode)
        rew, info = minion(filename, file_key=file_key, vid=args.vid)
        pprint.pprint(rew)
        pprint.pprint(info)


if __name__ == '__main__':
    main()
