#!/usr/bin/env python
#
# File: waterworld.py
#
# Created: Wednesday, August 24 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import argparse
import os
import yaml
import shutil
import sys
from rltools import util
from pipelines import pipeline

# Fix python 2.x
try:
    input = raw_input
except NameError:
    pass


def phase_train(spec, spec_file, git_hash):
    util.header('=== Running {} ==='.format(spec_file))

    # Make checkpoint dir. All outputs go here
    storagedir = spec['options']['storagedir']
    n_workers = spec['options']['n_workers']
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    util.mkdir_p(checkptdir)
    assert not os.listdir(checkptdir), 'Checkpoint directory {} is not empty!'.format(checkptdir)

    cmd_templates, output_filenames, argdicts = [], [], []
    train_spec = spec['training']
    arg_spec = spec['arguments']
    for alg in train_spec['algorithms']:
        for bline in train_spec['baselines']:
            for parch in train_spec['policy_archs']:
                for barch in train_spec['baseline_archs']:
                    for rad in arg_spec['radius']:
                        for n_se in arg_spec['n_sensors']:
                            for srange in arg_spec['sensor_ranges']:
                                for n_ev in arg_spec['n_evaders']:
                                    for n_pu in arg_spec['n_pursuers']:
                                        for n_co in arg_spec['n_coop']:
                                            if n_co > n_pu:
                                                continue
                                            for n_po in arg_spec['n_poison']:
                                                for f_rew in arg_spec['food_reward']:
                                                    for p_rew in arg_spec['poison_reward']:
                                                        for e_rew in arg_spec['encounter_reward']:
                                                            for disc in arg_spec['discounts']:
                                                                for gae in arg_spec['gae_lambdas']:
                                                                    for run in range(train_spec[
                                                                            'runs']):
                                                                        strid = (
                                                                            'alg={},bline={},parch={},barch={},'.
                                                                            format(alg['name'],
                                                                                   bline, parch,
                                                                                   barch) +
                                                                            'rad={},n_se={},srange={},n_ev={},n_pu={},n_co={},n_po={},'.
                                                                            format(rad, n_se,
                                                                                   srange, n_ev,
                                                                                   n_pu, n_co, n_po)
                                                                            +
                                                                            'f_rew={},p_rew={},e_rew={},'.
                                                                            format(f_rew, p_rew,
                                                                                   e_rew) +
                                                                            'disc={},gae={},run={}'.
                                                                            format(disc, gae, run))
                                                                        cmd_templates.append(alg[
                                                                            'cmd'].replace(
                                                                                '\n', ' ').strip())
                                                                        output_filenames.append(
                                                                            strid + '.txt')
                                                                        argdicts.append({
                                                                            'baseline_type': bline,
                                                                            'radius': rad,
                                                                            'sensor_range': srange,
                                                                            'n_sensors': n_se,
                                                                            'n_pursuers': n_pu,
                                                                            'n_evaders': n_ev,
                                                                            'n_coop': n_co,
                                                                            'n_poison': n_po,
                                                                            'discount': disc,
                                                                            'food_reward': f_rew,
                                                                            'poison_reward': p_rew,
                                                                            'encounter_reward':
                                                                                e_rew,
                                                                            'gae_lambda': gae,
                                                                            'policy_arch': parch,
                                                                            'baseline_arch': barch,
                                                                            'log': os.path.join(
                                                                                checkptdir,
                                                                                strid + '.h5')
                                                                        })

    util.ok('{} jobs to run...'.format(len(cmd_templates)))
    util.warn('Continue? y/n')
    if input() == 'y':
        pipeline.run_jobs(cmd_templates, output_filenames, argdicts, storagedir,
                          jobname=os.path.split(spec_file)[-1], n_workers=n_workers)
        sys.exit(0)
    else:
        util.failure('Canceled.')
        sys.exit(1)

    # Copy the pipeline yaml file to the output dir too
    shutil.copyfile(spec_file, os.path.join(checkptdir, 'pipeline.yaml'))
    with open(os.path.join(checkptdir, 'git_hash.txt'), 'w') as f:
        f.write(git_hash + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str)
    parser.add_argument('git_hash', type=str)
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = yaml.load(f)

    if args.git_hash is None:
        args.git_hash = '000000'
    phase_train(spec, args.spec, args.git_hash)


if __name__ == '__main__':
    main()
