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
import joblib
import os.path as osp
import uuid
import os

from gym import spaces
import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout, decrollout

from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_file', type=str) 
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--map_file', type=str, default='')
    args = parser.parse_args()

    policy_dir = osp.dirname(args.policy_file)
    params_file = osp.join(policy_dir, 'params.json')

    # Load file
    with open(params_file) as data_file:
        train_args = json.load(data_file)
    print('Loading parameters from {} in {}'.format(policy_dir, 'params.json'))


    with tf.Session() as sess:

        data = joblib.load(args.policy_file)

        policy = data['policy']
        env = data['env']

        if train_args['control'] == 'centralized':
            paths = rollout(env, policy, max_path_length=args.n_steps, animated=True)
        elif train_args['control'] == 'decentralized':
            paths = decrollout(env, policy, max_path_length=args.n_steps, animated=True)


    """
    if train_args['control'] == 'centralized':
        act_fn = lambda o: policy.get_action(o)[0]
    elif train_args['control'] == 'decentralized':
        def act_fn(o):
            action_list = []
            for agent_obs in o:
                a, adist = policy.get_action(agent_obs)
                action_list.append(a)
            return action_list
    env.animate(act_fn=act_fn, nsteps=args.n_steps, file_name=args.vid, verbose=args.verbose)
    """

if __name__ == '__main__':
    main()
