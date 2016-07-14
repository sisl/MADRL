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
from madrl_environments.pursuit.centralized_pursuit_evade import \
    CentralizedPursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from rltools.baseline import LinearFeatureBaseline, MLPBaseline, ZeroBaseline
from rltools.categorical_policy import CategoricalMLPPolicy
from rltools.sampler import ImportanceWeightedSampler, SimpleSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str) # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    args = parser.parse_args()

    # Load file
    filename, file_key = rltools.util.split_h5_name(args.filename)
    print('Loading parameters from {} in {}'.format(file_key, filename))
    with h5py.File(filename, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[file_key]

        pprint.pprint(dict(dset.attrs))

    pprint.pprint(train_args)
    env = CentralizedPursuitEvade(TwoDMaps.rectangle_map(*map(int, train_args['rectangle'].split(','))),
                                  n_evaders=train_args['n_evaders'],
                                  n_pursuers=train_args['n_pursuers'],
                                  obs_range=train_args['obs_range'],
                                  n_catch=train_args['n_catch'])

    policy = CategoricalMLPPolicy(env.observation_space, env.action_space,
                                  hidden_spec=train_args['policy_hidden_spec'],
                                  enable_obsnorm=True,
                                  tblog=train_args['tblog'], varscope_name='catmlp_policy')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        policy.load_h5(sess, filename, file_key)
        
        env.animate(act_fn=lambda o: policy.sample_actions(sess, o[None,...]), nsteps=200, file_name=args.vid)
        
if __name__ == '__main__':
    main()
