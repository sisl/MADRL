#!/usr/bin/env python
#
# File: main.py
#
# Created: Wednesday, July  6 2016 by rejuvyesh <mail@rejuvyesh.com>
#

import numpy as np
import tensorflow as tf

import gym
import policy

def main():
    env = gym.make('CartPole-v0')
    discount = 0.95
    # TODO SETUP
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        learner.train(sess)

        
if __name__ == '__main__':
    main()
