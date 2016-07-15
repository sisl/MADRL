#!/usr/bin/env python
#
# File: disc_pipeline.py
#
# Created: Friday, July 15 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import sys
sys.path.append('../rltools/')
import argparse
import os
import shutil
import subprocess

import yaml
import rltools.util
import pipeline

# Fix python 2.x
try:
    input = raw_input
except NameError:
    pass

def phase_train(spec, spec_file):
    rltools.util.header('=== Running {} ==='.format(spec_file))

    # Make checkpoint dir. All outputs go here
    storagedir = spec['options']['storagedir']
    n_workers = spec['options']['n_workers']
    checkptdir = os.path.join(spec['options']['storagedir'], spec['options']['checkpt_subdir'])
    rltools.util.mkdir_p(checkptdir)
    assert not os.listdir(checkptdir), 'Checkpoint directory {} is not empty!'.format(checkptdir)

    cmd_templates, output_filenames, argdicts = [], [], []
    for alg in spec['training']['algorithms']:
        for bline in spec['training']['baselines']:
            for rect in spec['rectangles']:
                for n_ev in spec['n_evaders']:
                    for n_pu in spec['n_pursuers']:
                        for orng in spec['obs_ranges']:
                            # observation range can't be bigger than the board
                            if orng > max(map(int, rect.split(','))):
                                continue
                            for n_ca in spec['n_catches']:
                                # number of simulataneous catches can't be bigger than numer of pusuers
                                if n_ca > n_pu:
                                    continue
                                for disc in spec['discounts']:
                                    for gae in spec['gae_lambdas']:
                                        for run in range(spec['training']['runs']):
                                            strid = 'alg={},bline={},rect={},n_ev={},n_pu={},orng={},n_ca={},disc={},gae={},run={}'.format(alg['name'], bline, rect, n_ev, n_pu, orng, n_ca, disc, gae, run)
                                            cmd_templates.append(alg['cmd'].replace('\n', ' ').strip())
                                            output_filenames.append(strid + '.txt')
                                            argdicts.append({
                                                'baseline_type': bline,
                                                'rectangle': rect,
                                                'n_evaders': n_ev,
                                                'n_pursuers': n_pu,
                                                'obs_range': orng,
                                                'n_catch': n_ca,
                                                'discount': disc,
                                                'gae_lambda': gae,
                                                'log': os.path.join(checkptdir, strid+'.h5')
                                            })

    rltools.util.ok('{} jobs to run...'.format(len(cmd_templates)))
    rltools.util.warn('Continue? y/n')
    if input() == 'y':
        pipeline.run_jobs(cmd_templates, output_filenames, argdicts, storagedir, n_workers=n_workers)
    else:
        rltools.util.failure('Canceled.')
        sys.exit(1)

    # Copy the pipeline yaml file to the output dir too
    shutil.copyfile(spec_file, os.path.join(checkptdir, 'pipeline.yaml'))
    # Keep git commit
    git_hash = subprocess.check_output('git rev-parse HEAD', shell=True).strip()
    with open(os.path.join(checkptdir, 'git_hash.txt'), 'w') as f:
        f.write(git_hash + '\n')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str)
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = yaml.load(f)

    phase_train(spec, args.spec)

if __name__ == '__main__':
    main()

