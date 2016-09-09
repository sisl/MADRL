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
import sys
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


def create_slurm_script(commands, outputfiles, jobname=None, nodes=4, cpus=6):
    assert len(commands) == len(outputfiles)
    template = '''#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name={jobname}
#################  
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=8:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes={nodes}
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=1 --cpus-per-task={cpus}
#################
module load singularity
export PYTHONPATH=/scratch/PI/mykel/src/python/rltools/:$PI_SCRATCH/src/python/rllab:$PI_SCRATCH/src/python/MADRL:$PYTHONPATH

read -r -d '' COMMANDS << END
{cmds_str}
END
cmd=$(echo "$COMMANDS" | awk "NR == $SLURM_ARRAY_TASK_ID")
echo $cmd

read -r -d '' OUTPUTFILES << END
{outputfiles_str}
END
outputfile=$SLURM_SUBMIT_DIR/$(echo "$OUTPUTFILES" | awk "NR == $SLURM_ARRAY_TASK_ID")
echo $outputfile
# Make sure output directory exists
mkdir -p "`dirname \"$outputfile\"`" 2>/dev/null

echo $cmd >$outputfile
eval $cmd >>$outputfile 2>&1
'''
    return template.format(jobname=jobname, nodes=nodes, cpus=cpus, cmds_str='\n'.join(commands),
                           outputfiles_str='\n'.join(outputfiles))


def run_slurm(cmd_templates, output_filenames, argdicts, storage_dir, outputfile_dir=None,
              jobname=None, n_workers=4, slurm_script_copy=None):
    assert len(cmd_templates) == len(output_filenames) == len(argdicts)
    num_cmds = len(cmd_templates)

    outputfile_dir = outputfile_dir if outputfile_dir is not None else 'logs_%s_%s' % (
        jobname, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    cmds, outputfiles = [], []
    for i in range(num_cmds):
        cmds.append(cmd_templates[i].format(**argdicts[i]))
        # outputfile_name = outputfile_prefixes[i] + ','.join('{}={}'.format(k,v) for k,v in sorted(argdicts[i].items())) + outputfile_suffix
        outputfiles.append(
            os.path.join(outputfile_dir, '{:04d}_{}'.format(i + 1, output_filenames[i])))

    script = create_slurm_script(cmds, outputfiles, jobname, nodes=n_workers)
    print(script)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.sh') as f:
        f.write(script)
        f.flush()

        cmd = 'sbatch --array %d-%d %s' % (1, len(cmds), f.name)
        print('Running command: {}'.format(cmd))
        print('ok ({} jobs)? y/n'.format(num_cmds))
        if raw_input() == 'y':
            # Write a copy of the script
            if slurm_script_copy is not None:
                assert not os.path.exists(slurm_script_copy)
                with open(slurm_script_copy, 'w') as fcopy:
                    fcopy.write(script)
                    print('slurm script written to {}'.format(slurm_script_copy))
            # Run slurm
            subprocess.check_call(cmd, shell=True)
        else:
            raise RuntimeError('Canceled.')


def eval_snapshot():
    pass


def eval_heuristic_for_snapshot():
    pass
