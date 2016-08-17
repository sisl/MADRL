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
from gym import spaces
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util

from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps

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
    if train_args['map_type'] == 'rectangle':
        env_map = TwoDMaps.rectangle_map(*map(int, train_args['rectangle'].split(',')))
    elif train_args['map_type'] == 'complex':
        env_map = TwoDMaps.complex_map(*map(int, train_args['rectangle'].split(',')))
    else:
        raise NotImplementedError()

    env = PursuitEvade(env_map,
                       n_evaders=train_args['n_evaders'],
                       n_pursuers=train_args['n_pursuers'],
                       obs_range=train_args['obs_range'],
                       n_catch=train_args['n_catch'],
                       urgency_reward=train_args['urgency'],
                       surround=train_args['surround'])

    if train_args['control'] == 'decentralized':
        obsfeat_space = env.agents[0].observation_space
        action_space = env.agents[0].action_space
    elif train_args['control'] == 'centralized':
        obsfeat_space = spaces.Box(low=env.agents[0].observation_space.low[0],
                                   high=env.agents[0].observation_space.high[0],
                                   shape=(env.agents[0].observation_space.shape[0] *
                                          len(env.agents),))  # XXX
        action_space = spaces.Discrete(env.agents[0].action_space.n * len(env.agents))

    else:
        raise NotImplementedError()


    policy = CategoricalMLPPolicy(obsfeat_space, action_space,
                                  hidden_spec=train_args['policy_hidden_spec'],
                                  enable_obsnorm=True,
                                  tblog=train_args['tblog'], varscope_name='pursuit_catmlp_policy')


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        policy.load_h5(sess, filename, file_key)
        if train_args['control'] == 'centralized':
            act_fn = lambda o: policy.sample_actions(sess, np.expand_dims(np.array(o).flatten(),0), deterministic=args.deterministic)[0][0,0]
        elif train_args['control'] == 'decentralized':
            def act_fn(o):
                action_list = []
                for agent_obs in o:
                    a, adist = policy.sample_actions(sess, np.expand_dims(agent_obs,0), deterministic=args.deterministic)
                    action_list.append(a[0, 0])
                return action_list
        #import IPython
        #IPython.embed()
        env.animate(act_fn=act_fn, nsteps=200, file_name=args.vid)

if __name__ == '__main__':
    main()
