#!/usr/bin/env python
#
# File: pipeline.py
#
# Created: Tuesday, July 12 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import sys
sys.path.append('../rltools/')

import argparse
import datetime
import multiprocessing as mp
import os
import shutil
import subprocess

import yaml

import rltools.util

# Fix python 2.x
try:
    input = raw_input
except NameError:
    pass

def runcommand(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).strip()
    except:
        return "Error executing command {}".format(cmd)


class Worker(mp.Process):

    def __init__(self, work_queue, result_queue):
        # base class initialization
        mp.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False

    def run(self):
        while (not (self.kill_received)) and (self.work_queue.empty()==False):
            try:
                job = self.work_queue.get_nowait()
                outfile = self.result_queue.get_nowait()
            except:
                break

            rtn_val = runcommand(job)
            with open(outfile, 'w') as f:
                f.write(rtn_val + '\n')


def run_jobs(cmd_templates, output_filenames, argdicts, outputfile_dir=None, jobname=None):
    assert len(cmd_templates) == len(output_filenames) == len(argdicts)
    num_cmds = len(cmd_templates)
    outputfile_dir = outputfile_dir if outputfile_dir is not None else 'logs_%s_%s' % (jobname, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    n_workers = mp.cpu_count() // 2

    cmds, outputfiles = [], []
    for i in range(num_cmds):
        cmds.append(cmd_templates[i].format(**argdicts[i]))
        outputfiles.append(os.path.join(outputfile_dir, '{:04d}_{}'.format(i+1, output_filenames[i])))

    work_queue = mp.Queue()
    res_queue = mp.Queue()
    for cmd, ofile in zip(cmds, outputfiles):
        work_queue.put(cmd)
        res_queue.put(ofile)

    worker = []
    for i in range(n_workers):
        worker.append(Worker(work_queue, res_queue))
        worker[i].start()


def phase_train(spec, spec_file):
    rltools.util.header('=== Running {} ==='.format(spec_file))

    # Make checkpoint dir. All outputs go here
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
                                            strid = 'alg={},bline={},rect={},n_ev={},n_pu={},orng={},n_ca={},disc={},run={}'.format(alg['name'], bline, rect, n_ev, n_pu, orng, n_ca, disc, run)
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
        run_jobs(cmd_templates, output_filenames, argdicts)
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
