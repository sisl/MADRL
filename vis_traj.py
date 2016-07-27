#!/usr/bin/env python
#
# File: vis_traj.py
#
# Created: Wednesday, July 13 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import pprint
import sys
sys.path.append('../rltools/')

import gym
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util
from madrl_environments.pursuit import CentralizedPursuitEvade, DecPursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.categorical import CategoricalMLPPolicy

from pursuit_policy import PursuitCentralMLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str) # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    args = parser.parse_args()

    # Load file
    filename, file_key = rltools.util.split_h5_name(args.filename)
    print('Loading parameters from {} in {}'.format(file_key, filename))
    with h5py.File(filename, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[file_key]

        pprint.pprint(dict(dset.attrs))

    pprint.pprint(train_args)
    if train_args['control'] == 'decentralized':
        env = DecPursuitEvade(TwoDMaps.rectangle_map(*map(int, train_args['rectangle'].split(','))),
                                      n_evaders=train_args['n_evaders'],
                                      n_pursuers=train_args['n_pursuers'],
                                      obs_range=train_args['obs_range'],
                                      n_catch=train_args['n_catch'])
    elif train_args['control'] == 'centralized':
        env = CentralizedPursuitEvade(TwoDMaps.rectangle_map(*map(int, train_args['rectangle'].split(','))),
                                      n_evaders=train_args['n_evaders'],
                                      n_pursuers=train_args['n_pursuers'],
                                      obs_range=train_args['obs_range'],
                                      n_catch=train_args['n_catch'])
    else:
        raise NotImplementedError()


    pursuit_policy = CategoricalMLPPolicy(env.observation_space, env.action_space,
                                  hidden_spec=train_args['policy_hidden_spec'],
                                  enable_obsnorm=True,
                                  tblog=train_args['tblog'], varscope_name='pursuit_catmlp_policy')

    evade_policy = CategoricalMLPPolicy(env.observation_space, env.action_space,
                                  hidden_spec=train_args['policy_hidden_spec'],
                                  enable_obsnorm=True,
                                  tblog=train_args['tblog'], varscope_name='evade_catmlp_policy')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        pursuit_policy.load_h5(sess, filename, file_key)
        #import IPython
        #IPython.embed()
        env.animate(act_fn=lambda o: pursuit_policy.sample_actions(sess, np.expand_dims(o,0), deterministic=args.deterministic), nsteps=200, file_name=args.vid)

if __name__ == '__main__':
    main()
