#!/usr/bin/env python
#
# File: pipeline.py
#
# Created: Tuesday, July 12 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import argparse
import datetime
import multiprocessing as mp
import os
import shutil
import subprocess

import yaml

import rltools.util


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
        while (not (self.kill_received)) and (self.work_queue.empty() == False):
            try:
                job = self.work_queue.get_nowait()
                outfile = self.result_queue.get_nowait()
            except:
                break
            print('Starting job: {}'.format(job))
            rtn_val = runcommand(job)
            with open(outfile, 'w') as f:
                f.write(rtn_val + '\n')


def run_jobs(cmd_templates, output_filenames, argdicts, storage_dir, outputfile_dir=None,
             jobname=None, n_workers=4):
    assert len(cmd_templates) == len(output_filenames) == len(argdicts)
    num_cmds = len(cmd_templates)
    outputfile_dir = outputfile_dir if outputfile_dir is not None else 'logs_%s_%s' % (
        jobname, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    rltools.util.mkdir_p(os.path.join(storage_dir, outputfile_dir))

    cmds, outputfiles = [], []
    for i in range(num_cmds):
        cmds.append(cmd_templates[i].format(**argdicts[i]))
        outputfiles.append(
            os.path.join(storage_dir, outputfile_dir, '{:04d}_{}'.format(i + 1, output_filenames[
                i])))

    work_queue = mp.Queue()
    res_queue = mp.Queue()
    for cmd, ofile in zip(cmds, outputfiles):
        work_queue.put(cmd)
        res_queue.put(ofile)

    worker = []
    for i in range(n_workers):
        worker.append(Worker(work_queue, res_queue))
        worker[i].start()
