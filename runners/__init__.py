from collections import namedtuple

import argparse
import sys
import datetime
import dateutil
import uuid
import ast

import archs


def tonamedtuple(dictionary):
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            dictionary[key] = tonamedtuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def get_arch(name):
    constructor = getattr(archs, name)
    return constructor


def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []


class RunnerParser(object):

    DEFAULT_OPTS = [
        ('discount', float, 0.95, ''),
        ('gae_lambda', float, 0.99, ''),
        ('n_iter', int, 500, ''),
    ]

    DEFAULT_POLICY_OPTS = [
        ('control', str, None, ''),
        ('recurrent', str, None, ''),
        ('baseline_type', str, 'linear', ''),
    ]

    def __init__(self, env_options, **kwargs):
        self._env_options = env_options
        parser = argparse.ArgumentParser(description='Runner')

        parser.add_argument('mode', help='rllab or rltools')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.mode):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        self._mode = args.mode
        getattr(self, args.mode)(self._env_options, **kwargs)

    def update_argument_parser(self, parser, options, **kwargs):
        kwargs = kwargs.copy()
        for (name, typ, default, desc) in options:
            flag = "--" + name
            if flag in parser._option_string_actions.keys():  #pylint: disable=W0212
                print("warning: already have option %s. skipping" % name)
            else:
                parser.add_argument(flag, type=typ, default=kwargs.pop(name, default), help=desc or
                                    " ")
        if kwargs:
            raise ValueError("options %s ignored" % kwargs)

    def rllab(self, env_options, **kwargs):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        rand_id = str(uuid.uuid4())[:5]
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, default=default_exp_name)
        self.update_argument_parser(parser, self.DEFAULT_OPTS)
        self.update_argument_parser(parser, self.DEFAULT_POLICY_OPTS)

        parser.add_argument(
            '--algo', type=str, default='tftrpo',
            help='Add tf or th to the algo name to run tensorflow or theano version')

        parser.add_argument('--max_path_length', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=12000)
        parser.add_argument('--n_parallel', type=int, default=1)

        parser.add_argument('--epoch_length', type=int, default=1000)
        parser.add_argument('--min_pool_size', type=int, default=10000)
        parser.add_argument('--qfunc_lr', type=float, default=1e-3)
        parser.add_argument('--policy_lr', type=float, default=1e-4)

        parser.add_argument('--feature_net', type=str, default=None)
        parser.add_argument('--feature_output', type=int, default=16)
        parser.add_argument('--feature_hidden', type=comma_sep_ints, default='128,64,32')
        parser.add_argument('--policy_hidden', type=comma_sep_ints, default='32')
        parser.add_argument('--min_std', type=float, default=1e-6)

        parser.add_argument('--step_size', type=float, default=0.01, help='max kl wall limit')

        parser.add_argument('--log_dir', type=str, required=False)
        parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                            help='Name of the tabular log file (in csv).')
        parser.add_argument('--text_log_file', type=str, default='debug.log',
                            help='Name of the text log file (in pure text).')
        parser.add_argument('--params_log_file', type=str, default='params.json',
                            help='Name of the parameter log file (in json).')
        parser.add_argument('--seed', type=int, help='Random seed for numpy')
        parser.add_argument('--args_data', type=str, help='Pickled data for stub objects')
        parser.add_argument('--snapshot_mode', type=str, default='all',
                            help='Mode to save the snapshot. Can be either "all" '
                            '(all iterations will be saved), "last" (only '
                            'the last iteration will be saved), or "none" '
                            '(do not save snapshots)')
        parser.add_argument(
            '--log_tabular_only', type=ast.literal_eval, default=False,
            help='Whether to only print the tabular log information (in a horizontal format)')

        self.update_argument_parser(parser, env_options, **kwargs)
        self.args = parser.parse_known_args(
            [arg for arg in sys.argv[2:] if arg not in ('-h', '--help')])[0]

    def rltools(self, env_options, **kwargs):
        parser = argparse.ArgumentParser()
        self.update_argument_parser(parser, self.DEFAULT_OPTS)
        self.update_argument_parser(parser, self.DEFAULT_POLICY_OPTS)

        parser.add_argument('--sampler', type=str, default='simple')
        parser.add_argument('--sampler_workers', type=int, default=1)
        parser.add_argument('--max_traj_len', type=int, default=500)
        parser.add_argument('--n_timesteps', type=int, default=12000)

        parser.add_argument('--adaptive_batch', action='store_true', default=False)
        parser.add_argument('--n_timesteps_min', type=int, default=4000)
        parser.add_argument('--n_timesteps_max', type=int, default=64000)
        parser.add_argument('--timestep_rate', type=int, default=20)

        parser.add_argument('--policy_hidden_spec', type=get_arch, default='GAE_ARCH')
        parser.add_argument('--baseline_hidden_spec', type=get_arch, default='GAE_ARCH')
        parser.add_argument('--min_std', type=float, default=1e-6)
        parser.add_argument('--max_kl', type=float, default=0.01)
        parser.add_argument('--vf_max_kl', type=float, default=0.01)
        parser.add_argument('--vf_cg_damping', type=float, default=0.01)
        parser.add_argument('--enable_obsnorm', action='store_true')
        parser.add_argument('--enable_rewnorm', action='store_true')
        parser.add_argument('--enable_vnorm', action='store_true')

        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--log', type=str, required=False)
        parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb_{}'.format(uuid.uuid4()))
        parser.add_argument('--no-debug', dest='debug', action='store_false')
        parser.set_defaults(debug=True)
        self.update_argument_parser(parser, env_options, **kwargs)
        self.args = parser.parse_known_args(
            [arg for arg in sys.argv[2:] if arg not in ('-h', '--help')])[0]
