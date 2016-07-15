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
import sys
sys.path.append('../rltools/')

import gym
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util
from madrl_environments.pursuit import CentralizedWaterWorld
from rltools.baseline import LinearFeatureBaseline, MLPBaseline, ZeroBaseline
from rltools.gaussian_policy import GaussianMLPPolicy

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

    env = CentralizedWaterWorld(train_args['n_pursuers'], train_args['n_evaders'], train_args['n_coop'], train_args['n_poison'], n_sensors=train_args['n_sensors'])

    policy = GaussianMLPPolicy(env.observation_space, env.action_space, hidden_spec=train_args['policy_hidden_spec'],
                               enable_obsnorm=True,
                               min_stdev=0.,
                               init_logstdev=0.,
                               tblog=train_args['tblog'],
                               varscope_name='gaussmlp_policy')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        policy.load_h5(sess, filename, file_key)

        rew = env.animate(act_fn=lambda o: policy.sample_actions(sess, o[None,...], deterministic=True), nsteps=500, file_name=args.vid)
        print(rew)

if __name__ == '__main__':
    main()
