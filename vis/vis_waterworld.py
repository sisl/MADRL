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

from gym import spaces
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import MAWaterWorld
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--centralized', action='store_true', default=False)
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

    centralized = True if train_args['control'] == 'centralized' else False
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

    if centralized:
        obsfeat_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                                   high=env.agents[0].observation_space.high[0],
                                   shape=(env.agents[0].observation_space.shape[0] *
                                          len(env.agents),))  # XXX
        action_space = spaces.Box(low=env.agents[0].action_space.low[0],
                                  high=env.agents[0].action_space.high[0],
                                  shape=(env.agents[0].action_space.shape[0] *
                                         len(env.agents),))  # XXX
    else:
        obsfeat_space = env.agents[0].observation_space
        action_space = env.agents[0].action_space

    policy = GaussianMLPPolicy(env.observation_space, env.action_space,
                               hidden_spec=train_args['policy_hidden_spec'], enable_obsnorm=True,
                               min_stdev=0., init_logstdev=0., tblog=train_args['tblog'],
                               varscope_name='gaussmlp_policy')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        policy.load_h5(sess, filename, file_key)

        rew = env.animate(
            act_fn=lambda o: policy.sample_actions(sess, o[None, ...], deterministic=args.deterministic),
            nsteps=args.n_steps, file_name=args.vid)
        print(rew)


if __name__ == '__main__':
    main()
