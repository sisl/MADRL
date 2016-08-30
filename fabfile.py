#!/usr/bin/env python
#
# File: fabfile.py
#
# Created: Wednesday, August 24 2016 by rejuvyesh <mail@rejuvyesh.com>
# License: GNU GPL 3 <http://www.gnu.org/copyleft/gpl.html>
#
from fabric.api import cd, put, path, task, shell_env, run, env, local, settings
from fabric.contrib.project import rsync_project
import os.path
from time import sleep

env.use_ssh_config = True

RLTOOLS_LOC = '/home/{}/src/python/rltools'.format(env.user)
MADRL_LOC = '/home/{}/src/python/MADRL'.format(env.user)


class Tmux(object):

    def __init__(self, name):
        self._name = name
        with settings(warn_only=True):
            test = run('tmux has-session -t {}'.format(self._name))
        if test.failed:
            run('tmux new-session -d -s {}'.format(self._name))

    def run(self, cmd, window=0):
        run('tmux send -t {}.{} "{}" ENTER'.format(self._name, window, cmd))


@task
def githash():
    git_hash = local('git rev-parse HEAD', capture=True)
    return git_hash

@task
def sync():
    rsync_project(remote_dir=os.path.split(MADRL_LOC)[0], exclude=['*.h5'])

@task(alias='pipe')
def runpipeline(script, fname):
    git_hash = githash()
    sync()
    pipetm = Tmux('pipeline')
    pipetm.run('export PYTHONPATH={}:{}'.format(RLTOOLS_LOC, MADRL_LOC))
    pipetm.run('cd {}'.format(MADRL_LOC))
    pipetm.run('python {} {} {}'.format(script, fname, git_hash))
    sleep(0.5)
    pipetm.run('y')
