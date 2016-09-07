#!/usr/bin/env python
#
# File: run_pipeline.py
#
# Created: Tuesday, September  6 2016 by rejuvyesh <mail@rejuvyesh.com>
import argparse
import os
import shutil
import sys
import yaml
import pandas as pd
import numpy as np

import rltools.util
from pipelines.pipeline import run_jobs, eval_snapshot, eval_heuristic_for_snapshot
# Fix python 2.x
try:
    input = raw_input
except NameError:
    pass


def phase1_train(spec, spec_file, git_hash, n_workers=2):
    rltools.util.header('==== Phase 1: training {} ===='.format(spec_file))

    # all outputs go here
    storagedir = spec['options']['storagedir']
    checkptdir = os.path.join(storagedir, spec['options']['checkpt_subdir'])
    rltools.util.mkdir_p(checkptdir)

    # Make sure dir is empty
    assert not os.listdir(checkptdir), 'Checkpint directory {} is not empty!'.format(checkptdir)

    sample_workers = spec['options']['sample_workers']

    # Assemble the commands to run
    cmd_templates, output_filenames, argdicts = [], [], []
    for alg in spec['training']['algorithms']:
        agent_arg_list = map(dict, zip(
            *[[(k, v) for v in value] for k, value in spec['training']['agents'].items()]))
        for agent_args in agent_arg_list:
            agent_id = ''
            for k, v in agent_args.items():
                agent_id += '{}={},'.format(k, v)
            for run in range(spec['training']['runs']):
                # string identifier
                strid = 'alg={},{}run={}'.format(alg['name'], agent_id, run)
                cmd_templates.append(alg['cmd'].replace('\n', ' ').strip())
                output_filenames.append(strid + '.txt')
                argdicts.append({'sample_workers': sample_workers,
                                 'log': os.path.join(checkptdir, strid + 'h5')}.update(agent_args))

    rltools.util.ok('{} jobs to run...'.format(len(cmd_templates)))
    rltools.util.warn('Continue? y/n')
    if input() == 'y':
        run_jobs(cmd_templates, output_filenames, argdicts, storagedir,
                 jobname=os.path.split(spec_file)[-1], n_workers=n_workers)
    else:
        rltools.util.failure('Canceled')
        sys.exit(1)

    shutil.copyfile(spec_file, os.path.join(checkptdir, 'pipeline.yaml'))
    with open(os.path.join(checkptdir, 'git_has.txt', 'w')) as f:
        f.write(git_hash + '\n')


def phase2_eval(spec, spec_file):
    rltools.util.header('==== Phase 2: evaluating trained models ====')
    envname = spec['task']['env']
    storagedir = spec['options']['storagedir']
    checkptdir = os.path.join(storagedir, spec['options']['checkpt_subdir'])
    print('Evluating results in {}'.format(checkptdir))

    results_full_path = os.path.join(storagedir, spec['options']['results_filename'])
    print('Results will be stored in {}'.format(results_full_path))
    if os.path.exists(results_full_path):
        raise RuntimeError('Results file {} already exists'.format(results_full_path))

    evals_to_do = []
    nonexistent_checkptfile = []
    for alg in spec['training']['algorithms']:
        agent_arg_list = map(dict, zip(
            *[[(k, v) for v in value] for k, value in spec['training']['agents'].items()]))
        for agent_args in agent_arg_list:
            agent_id = ''
            for k, v in agent_args.items():
                agent_id += '{}={},'.format(k, v)
            for run in range(spec['training']['runs']):
                # Make sure checkpoint file exists
                strid = 'alg={},{}run={}'.format(alg['name'], agent_id, run)
                checkptfile = os.path.join(checkptdir, strid, '.h5')
                if not os.path.exists(checkptfile):
                    nonexistent_checkptfile.append(checkptfile)

                evals_to_do.append((alg, agent_args, run, checkptfile))

    if nonexistent_checkptfile:
        print('Cannot find checkpoint files:')
        print('\n'.join(nonexistent_checkptfile))
        raise RuntimeError()

    # Walk through all saved checkpoints
    collected_results = []
    for i_eval, (alg, agent_args, run, checkptfile) in enumerate(evals_to_do):
        agent_id = ''
        for k, v in agent_args.items():
            agent_id += '{}={},'.format(k, v)
            rltools.util.header('Evaluating run {}/{} : alg={},{}run={}'.format(i_eval + 1, len(
                evals_to_do), alg['name'], agent_id, run))

        # Load checkpoint file
        with pd.HDFStore(checkptfile, 'r') as f:
            log_df = f['log']
            log_f.set_index('iter', inplace=True)

            # Evaluate return
            snapshot_names = f.root.snapshots._v_children.keys()
            assert all(name.startswith('iter') for name in snapshot_names)
            snapshot_inds = sorted([int(name[len('iter'):]) for name in snapshot_names])

            if 'eval_up_to_iter' in alg:
                snapshot_inds = [sidx for sidx in snapshot_inds if sidx <= alg['eval_up_to_iter']]
                print('Restricting snapshots for {} up to iter {}'.format(checkptfile, alg[
                    'eval_up_to_iter']))

            last_snapshot_idx = snapshot_inds[-1]
            ret, info = eval_snapshot(envname, checkptfile, last_snapshot_idx,
                                      spec['options']['eval_num_trajs'])

            hret, hinfo = eval_heuristic_for_snapshot(envname, checkptfile, last_snapshot_idx,
                                                      spec['options']['eval_num_trajs'])

            collected_results.append({
                'alg': alg['name'],
                'env': envname,
                'run': run,
                'ret': ret,
                'info': info,
                'hret': hret,
                'hinfo': hinfo,
            })

    collected_results = pd.DataFrame(collected_results)
    with pd.HDFStore(results_full_path, 'w') as outf:
        outf['results'] = collected_results


def phase3_csv():
    pass


def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)
    # Keep git commit
    import subprocess
    git_hash = subprocess.check_output('git rev-parse HEAD', shell=True).strip()

    phases = {'1_train': lambda *args: phase1_train(*args, git_hash=git_hash),
              '2_eval': phase2_eval,
              '3_csv': phase3_csv}
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=str)
    parser.add_argument('phase', choices=sorted(phases.keys()))
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = yaml.load(f)

    phases[args.phase](spec, args.spec)


if __name__ == '__main__':
    main()
