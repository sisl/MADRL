import uuid
import os
import json
import pprint
import h5py

import numpy as np
import tensorflow as tf
from gym import spaces

import rltools.util

from runners.rurllab import rllab_envpolicy_parser
from runners.rurltools import rltools_envpolicy_parser


class FileHandler(object):

    def __init__(self, filename):
        self.filename = filename
        # Handle remote files
        if ':' in filename:
            tmpfilename = str(uuid.uuid4())
            if 'h5' in filename.split('.')[-1]:
                os.system('rsync -avrz {}.h5 /tmp/{}.h5'.format(filename.split('.')[0],
                                                                tmpfilename))
                newfilename = '/tmp/{}.{}'.format(tmpfilename, filename.split('.')[-1])
                self.filename = newfilename
            else:
                os.system('rsync -avrz {} /tmp/{}.pkl'.format(filename, tmpfilename))
                newfilename = '/tmp/{}.pkl'.format(tmpfilename)
                self.filename = newfilename
                # json file?
                # TODO

            # Loading file
        if 'h5' in self.filename.split('.')[-1]:
            self.mode = 'rltools'
            self.filename, self.file_key = rltools.util.split_h5_name(self.filename)
            print('Loading parameters from {} in {}'.format(self.file_key, filename))
            with h5py.File(self.filename, 'r') as f:
                self.train_args = json.loads(f.attrs['args'])
                dset = f[self.file_key]
                pprint.pprint(dict(dset.attrs))

        else:
            self.mode = 'rllab'
            policy_dir = os.path.dirname(self.filename)
            params_file = os.path.join(policy_dir, 'params.json')
            self.filename = self.filename
            self.file_key = None
            print('Loading parameters from {} in {}'.format('params.json', policy_dir))
            with open(params_file, 'r') as df:
                self.train_args = json.load(df)


class PolicyLoad(object):

    def __init__(self, env, train_args, max_traj_len, n_trajs, deterministic, mode='rltools'):

        self.mode = mode
        if self.mode == 'rltools':
            self.env, self.policy = rltools_envpolicy_parser(env, train_args)
        elif self.mode == 'rllab':
            self.env, _ = rllab_envpolicy_parser(env, train_args)
            self.policy = None

        self.deterministic = deterministic
        self.max_traj_len = max_traj_len
        self.n_trajs = n_trajs
        self.disc = train_args['discount']
        self.control = train_args['control']


class Evaluator(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        if self.mode == 'rltools':
            file_key = kwargs.pop('file_key', None)
            assert file_key
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                self.policy.load_h5(sess, filename, file_key)
                return rltools.util.evaluate_policy(self.env, self.policy,
                                                    deterministic=self.deterministic,
                                                    disc=self.disc, mode=self.control,
                                                    max_traj_len=self.max_traj_len,
                                                    n_trajs=self.n_trajs)


class Visualizer(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Visualizer, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        if self.mode == 'rltools':
            file_key = kwargs.pop('file_key', None)
            assert file_key
            vid = kwargs.pop('vid', None)
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                self.policy.load_h5(sess, filename, file_key)
                rew, trajinfo = self.env.animate(
                    act_fn=lambda o: self.policy.sample_actions(o[None, ...], deterministic=self.deterministic)[0],
                    nsteps=self.max_traj_len)
                info = {key: np.sum(value) for key, value in trajinfo.items()}
                return (rew, info)

        if self.mode == 'rllab':
            import joblib
            from rllab.sampler.utils import rollout, decrollout

            # XXX
            tf.reset_default_graph()
            with tf.Session() as sess:

                data = joblib.load(filename)
                policy = data['policy']
                if self.control == 'centralized':
                    path = rollout(self.env, policy, max_path_length=self.max_traj_len,
                                   animated=True)
                    rew = path['rewards'].mean()
                    info = path['env_infos'].mean()
                elif self.control == 'decentralized':
                    paths = decrollout(self.env, policy, max_path_length=self.max_traj_len,
                                       animated=True)
                    rew = [path['rewards'].mean() for path in paths]
                    info = {key: value.sum() for key, value in paths[0]['env_infos'].items()}
                return rew, info
