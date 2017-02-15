import copy
import math
import sys
from lxml import etree
import lxml.builder    

import gym
import numpy as np
from gym import spaces

from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle


class Leg(Agent):

    def __init__(self, idx):
        self._idx = idx


class MultiAnt(AbstractMAEnv, EzPickle):

    def __init__(self, 
                 n_legs=4,
                 ts=0.02,
                 integrator='RK4',
                 ):
        assert n_legs % 2 == 0, "n_legs has to be even, passed in : %r" % n_legs
        self.n_legs = n_legs


        self.gen_xml()



    def gen_xml(self, out_file="ant.xml",
                      og_file="ant_og.xml"):
        """Write .xml file for the ant problem.
        Modify original 4 leg ant defintion.
        """

        parser = etree.XMLParser(remove_blank_text=True)
        og = etree.parse(og_file, parser)


        # add legs
        torso = og.find('.//body')
        # first remove original legs
        for c in torso.getchildren():
            if c.tag == 'body':
                torso.remove(c)
        for i in xrange(self.n_legs):
            etree.SubElement(torso, "body",
                                    name="leg_"+str(i),
                                    pos="0 0 0")

        # add new motors
        actuators = og.find('actuator')
        actuators.clear()
        for i in xrange(self.n_legs):
            etree.SubElement(actuators, "motor",
                                        joint="hip_"+str(i),
                                        ctrlrange="-150.0 150.0",
                                        ctrllimited="true")
            etree.SubElement(actuators, "motor",
                                        joint="ankle_"+str(i),
                                        ctrlrange="-150.0 150.0",
                                        ctrllimited="true")

        og.write(out_file, pretty_print=True)


    def get_point_on_circle(self, r, current_point, total_points):
        theta = 2*np.pi / total_points
        angle = theta * current_point
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y


if __name__ == '__main__':
    env = MultiAnt(10)
