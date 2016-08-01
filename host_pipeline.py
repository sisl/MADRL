#!/usr/bin/env python
#
# File: host_pipeline.py
#
# Created: Monday, August  1 2016 by rejuvyesh <mail@rejuvyesh.com>
# License: GNU GPL 3 <http://www.gnu.org/copyleft/gpl.html>
#
import sys
sys.path.append('../rltools')
import argparse
import os
import yaml
import shutil
import rltools
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
            for n_g in spec['n_good']:
                for n_h in spec['n_hostages']:
                    for n_b in spec['n_bad']:
                        for n_cs in spec['n_coop_save']:
                            if n_cs > n_g:
                                continue
                            for n_ca in spec['n_coop_avoid']:
                                if n_ca > n_g:
                                    continue
                                for n_se in spec['n_sensors']:
                                    for srange in spec['sensor_range']:
                                        for srew in spec['save_reward']:
                                            for hrew in spec['hit_reward']:
                                                for erew in spec['encounter_reward']:
                                                    for borew in spec['bomb_reward']:
                                                        for disc in spec['discounts']:
                                                            for gae in spec['gae_lambdas']:
                                                                for run in range(spec['training'][
                                                                        'runs']):
                                                                    strid = 'alg={},bline={},n_g={},n_h={},n_b={},n_cs={},n_ca={},n_se={},srange={},srew={},hrew={},erew={},borew={},disc={},gae={},run={}'.format(
                                                                        alg['name'], bline, n_g,
                                                                        n_h, n_b, n_cs, n_ca, n_se,
                                                                        srange, srew, hrew, erew,
                                                                        borew, disc, gae, run)
                                                                    cmd_templates.append(alg[
                                                                        'cmd'].replace('\n',
                                                                                       ' ').strip())
                                                                    output_filenames.append(strid +
                                                                                            '.txt')
                                                                    argdicts.append({
                                                                        'baseline_type': bline,
                                                                        'n_good': n_g,
                                                                        'n_hostage': n_h,
                                                                        'n_bad': n_b,
                                                                        'n_coop_save': n_cs,
                                                                        'n_coop_avoid': n_ca,
                                                                        'n_sensors': n_se,
                                                                        'sensor_range': srange,
                                                                        'save_reward': srew,
                                                                        'hit_reward': hrew,
                                                                        'encounter_reward': erew,
                                                                        'bomb_reward': borew,
                                                                        'discount': disc,
                                                                        'gae_lambda': gae,
                                                                        'log': os.path.join(
                                                                            checkptdir,
                                                                            strid + '.h5')
                                                                    })

    rltools.util.ok('{} jobs to run...'.format(len(cmd_templates)))
    rltools.util.warn('Continue? y/n')
    if input() == 'y':
        pipeline.run_jobs(cmd_templates, output_filenames, argdicts, storagedir,
                          n_workers=n_workers)
    else:
        rltools.util.failure('Canceled.')
        sys.exit(1)

    # Copy the pipeline yaml file to the output dir too
    shutil.copyfile(spec_file, os.path.join(checkptdir, 'pipeline.yaml'))
    # Keep git commit
    import subprocess
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
