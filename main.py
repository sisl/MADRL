#!/usr/bin/env python
#
# File: main.py
#
# Created: Wednesday, July  6 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

import log
import gym
import algos
from baseline import LinearFeatureBaseline
from categorical_policy import CategoricalMLPPolicy

def main():
    env = gym.make('CartPole-v0')
    discount = 0.95
    hidden_spec = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''
    
    policy = CategoricalMLPPolicy(env.observation_space, env.action_space, hidden_spec=hidden_spec, varscope_name='catmlp_policy')
    baseline = LinearFeatureBaseline(env.observation_space, env.action_space)
    step_func = algos.TRPO(max_kl=0.01)
    popt = algos.SamplingPolicyOptimizer(
        env=env,
        policy=policy,
        baseline=baseline,
        step_func=step_func,
        discount=discount
    )
    save_file = '/tmp/s.h5'
    log_f = log.TrainingLog(save_file, [])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f)

        
if __name__ == '__main__':
    main()
