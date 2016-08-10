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
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--n_steps', type=int, default=500)
    args = parser.parse_args()

    # Load file
    filename, file_key = rltools.util.split_h5_name(args.filename)
    print('Loading parameters from {} in {}'.format(file_key, filename))
    with h5py.File(filename, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[file_key]

        pprint.pprint(dict(dset.attrs))

    env = gym.make('BipedalWalker-v2')

    policy = GaussianMLPPolicy(env.observation_space, env.action_space,
                               hidden_spec=train_args['policy_hidden_spec'], enable_obsnorm=True,
                               min_stdev=0., init_logstdev=0., tblog=train_args['tblog'],
                               varscope_name='gaussmlp_policy')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        policy.load_h5(sess, filename, file_key)

        o = env.reset()
        for i in xrange(args.n_steps):
            env.render()
            a, adist = policy.sample_actions(sess, o[None,...], deterministic=args.deterministic)
            o, r, done, _ = env.step(a[0])
            print("Action:", a)
            print("Step:", o, r, done)
            if done:
                break

        """
        rew = env.animate(
            act_fn=lambda o: policy.sample_actions(sess, o[None, ...], deterministic=args.deterministic),
            nsteps=args.n_steps, file_name=args.vid)
        print(rew)
        """


if __name__ == '__main__':
    main()
