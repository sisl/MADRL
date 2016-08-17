import numpy as np
from collections import deque
import ode
from gym.utils import seeding
import vapory as vap
# Object constants
LENGTH = 0.6  # object's length
WIDTH = 0.2
HEIGHT = 0.05  # object's height
MASS = 1  # object's mass

# environment constants
MU = 0  #0.5      # the global mu to use # this parameter is discarded, use FRIC instead
GRAVITY = 9.81  #0.5  # the global gravity to use
FRIC = 5  # friction
MU_V = 0.2  # coefficient of the viscous force, which is proportional to velocity

# wall constants
nWall = 6
WALL_THICK = 0.04
WALL_TALL = 0.3

# robot constants
#nRobot = 12
FMAX = 1.4
TMAX = 2
ROBOT_RADIUS = 0.03  # used to visualize the robots, which are drew as spheres

# other constants
TIME_STEP = 0.01
TIME_INTERVAL = 50

WALL_LENGTH = np.array([3, 2.5, 3, 3, 2, 2.5])
WALL_POS = np.array([[5.5, 2], [5.25, 3], [7, 3.5], [6.5, 4.5], [8, 5], [7.75, 6]])
WALL_DIR = [0, 0, 1, 1, 0, 0]  # 1: y-axis wall


class OdeObj(object):

    def __new__(cls, *args, **kwargs):
        obj = super(OdeObj, cls).__new__(cls)
        return obj

    @property
    def body(self):
        raise NotImplementedError()

    @property
    def geom(self):
        raise NotImplementedError()

    @property
    def rendered(self):
        raise NotImplementedError()

    def setPos(self, *args):
        self.body.setPosition(args)

    def setQuat(self, *args):
        self.body.setQuaternion(args)

    def getPos(self):
        return self.body.getPosition()

    def getQuat(self):
        return self.body.getQuaternion()


class Box(OdeObj):

    def __init__(self, space, world, size, mass, color=None):
        self._size = size
        self._color = color
        assert len(size) == 3
        self._odebody = ode.Body(world)
        if mass:
            self._odemass = ode.Mass()
            self._odemass.setBox(1, *size)
            self._odemass.adjust(mass)
            self._odebody.setMass(self._odemass)

        self._odegeom = ode.GeomBox(space, size)
        self._odegeom.setBody(self._odebody)

    @property
    def body(self):
        return self._odebody

    @property
    def geom(self):
        return self._odegeom

    @property
    def rendered(self):
        return vap.Box(
            [-s / 2 for s in self._size], [s / 2 for s in self._size],
            vap.Texture('T_Ruby_Glass' if not self._color else vap.Pigment('color', self._color)),
            vap.Interior('ior', 4), 'matrix', self.body.getRotation() + self.body.getPosition())


class SphereRobot(OdeObj):

    def __init__(self, space, world, radius, mass, color=None):
        self._radius = radius
        self._color = color
        self._odemass = ode.Mass()
        self._odemass.setSphereTotal(0.00001, radius)
        self._odemass.adjust(mass)
        self._odebody = ode.Body(world)
        self._odebody.setMass(self._odemass)

        self._odegeom = ode.GeomSphere(space, radius)
        self._odegeom.setBody(self._odebody)

    @property
    def body(self):
        return self._odebody

    @property
    def geom(self):
        return self._odegeom

    @property
    def rendered(self):
        return vap.Sphere(
            list(self.body.getPosition()), self._radius,
            vap.Texture(vap.Pigment('color', self._color if self._color else [1, 0, 1])))


def axisangle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    angle /= 2
    x, y, z = axis
    w = np.cos(angle)
    x *= np.sin(angle)
    y *= np.sin(angle)
    z *= np.sin(angle)
    return [x, y, z, w]


class BoxPushing(object):

    def __init__(self, is_static):
        self._is_static = is_static
        self.nRobot = 12
        self.world = ode.World()
        self.world.setGravity((0, 0, -GRAVITY))
        self.space = ode.HashSpace()
        self.space.enable()
        self.ground = ode.GeomPlane(self.space, (0, 0, 1), 0)
        self.contactgroup = ode.JointGroup()

        self.obj = Box(self.space, self.world, (LENGTH, WIDTH, HEIGHT), MASS)

        self.wall = [None for _ in range(nWall)]

        self.robot = [None for _ in range(self.nRobot)]
        for i in range(self.nRobot):
            self.robot[i] = SphereRobot(self.space, self.world, ROBOT_RADIUS, MASS)

        self.joint = [None for _ in range(self.nRobot)]

        self.seed()

        self.objv = deque(maxlen=3)
        [self.objv.append(np.zeros(3)) for _ in range(3)]
        self.result_force = np.zeros(2)
        self.count = 0
        self.drift_count = 0
        self.sim_time = 0

    def _init_force(self):
        force_NR_2 = self.np_random.rand(self.nRobot, 2) * FMAX / 2
        return force_NR_2

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # ode.Util.randSetSeed(seed2)
        return [seed1]

    def reset(self):
        # Object
        self.obj.setPos(0, 0, 0)
        # Wall
        for i in range(nWall):
            self.wall[i] = Box(self.space, self.world, (WALL_LENGTH[i], WALL_THICK, WALL_TALL),
                               None, [0.3, 0.7, 0.1])
            self.wall[i].setPos(WALL_POS[i, 0], WALL_POS[i, 1], WALL_TALL / 2)
            if WALL_DIR[i] == 1:
                R = self.wall[i].getQuat()
                Q = axisangle_to_quat(np.array(list(R[:-1])), np.pi / 2)
                self.wall[i].setQuat(*Q)
        # Robots
        for i in range(4):
            self.robot[i].setPos(0.33 - i * 0.22, 0.13, ROBOT_RADIUS)
        for i in range(4, 8):
            self.robot[i].setPos(0.33 - (i - 4) * 0.22, -0.13, ROBOT_RADIUS)
        for i in range(8, 10):
            self.robot[i].setPos(0.33, 0.047 - (i - 8) * 0.087, ROBOT_RADIUS)
        for i in range(10, 12):
            self.robot[i].setPos(-0.33, 0.047 - (i - 10) * 0.087, ROBOT_RADIUS)

        for i in range(self.nRobot):
            self.joint[i] = ode.FixedJoint(self.world)
            self.joint[i].attach(self.obj.body, self.robot[i].body)
            self.joint[i].setFixed()

        force_NR_2 = self._init_force()
        self._add_force(force_NR_2)

    def _update_fric_dir(self):
        spdd = np.array(list(self.obj.body.getLinearVel()))
        if not self._is_static:
            self.fricdir = spdd / np.linalg.norm(spdd)
        else:
            self.fricdir = self.robot_sum_force / np.linalg.norm(self.robot_sum_force)

    def _add_force(self, force_NR_2):
        self.robot_sum_force = np.zeros(2)
        for i in range(self.nRobot):
            self.robot_sum_force += force_NR_2[i, :]

        self._update_fric_dir()

        if self._is_static:
            if np.linalg.norm(self.robot_sum_force) <= FRIC:
                if np.linalg.norm(self.objv[-1]) >= 0.07:
                    vel = self.objv[-1] / np.linalg.norm(self.objv[-1])
                    self.result_force = -FRIC * vel[:2]
                else:
                    self.result_force = np.zeros(2)
            else:
                self.result_force = self.robot_sum_force - FRIC * self.fricdir
        else:  # Dynamic friction
            self.result_force = self.robot_sum_force - FRIC * self.fricdir - MU_V * self.objv[-1][:
                                                                                                  2]

        self.obj.body.addForce((self.result_force[0], self.result_force[1], 0))

    def _get_acc(self):
        objv = np.array(self.objv)
        dv = objv[1:] - objv[:-1]
        acc = dv.mean() * 1 / TIME_STEP
        return acc

    def _near_callback(self, _, geom1, geom2):
        g1 = (geom1 == self.ground)
        g2 = (geom2 == self.ground)
        if not (g1 ^ g2):
            return

        b1 = geom1.getBody()
        b2 = geom2.getBody()

        contact = ode.collide(geom1, geom2)
        for con in contact[:3]:
            con.setMode(ode.ContactSoftCFM | ode.ContactApprox1)
            con.setMu(MU)
            con.setSoftCFM(0.01)
            j = ode.ContactJoint(self.world, self.contactgroup, con)
            j.attach(b1, b2)

    @property
    def is_terminal(self):
        pass

    def _info(self):
        pos = self.obj.getPos()
        print("-" * 20)
        print("Rbt Force = {}, sum = {}".format(self.robot_sum_force, np.linalg.norm(
            self.robot_sum_force)))
        print("End Force = {}, sum = {}".format(self.result_force, np.linalg.norm(
            self.result_force)))
        print("Pos: {}".format(pos))
        print("Vel: {}, Acc: {}".format(self.objv[-1], self.objacc))
        print("Abs Vel: {}".format(np.linalg.norm(self.objv[-1])))
        print("Simtime: {}".format(self.sim_time))

    def step(self, force_NR_2):
        self.count += 1
        self.sim_time += TIME_STEP
        self._add_force(force_NR_2)
        self.space.collide(None, self._near_callback)
        self.world.step(TIME_STEP)
        self.contactgroup.empty()

        speed = self.obj.body.getLinearVel()
        self.objv.append(np.array(list(speed)))
        self.objacc = self._get_acc()

        if self.count == TIME_INTERVAL:
            self._info()
            self.count = 0
            if any(self.objv[-1] == 0) and any(self.result_force == 0):
                self.drift_count += 1
            if self.drift_count == 2:
                self.drift_count = 0
                self.obj.body.setLinearVel((0, 0, 0))

    def render(self, screen_size):
        light = vap.LightSource([3, 3, 3], 'color', [3, 3, 3], 'parallel', 'point_at', [0, 0, 0])
        camera = vap.Camera('location', [0.5 * 10, -2 * 10, 3 * 10], 'look_at', [0, 0, 0])
        ground = vap.Plane([0, 0, 1], 0, vap.Texture('T_Stone33'))
        walls = [wall.rendered for wall in self.wall]
        robots = [bot.rendered for bot in self.robot]
        obj = self.obj.rendered
        scene = vap.Scene(camera, [light, ground, vap.Background("White"), obj] + robots + walls,
                          included=["colors.inc", "textures.inc", "glass.inc", "stones.inc"])
        return scene.render(height=screen_size, width=screen_size, antialiasing=0.0001)


if __name__ == '__main__':
    env = BoxPushing(is_static=True)
    env.reset()
    print('n:{}'.format(env.space.getNumGeoms()))
    print('g:{}'.format(env.world.getGravity()))
    print('o:{}'.format(env.obj.getPos()))
    for i in range(env.nRobot):
        print(env.robot[i].getPos())
        print('---')

    env.render(800)
    while True:
        env.step(env._init_force())
    # def make_frame(t):
    #     env.step(env._init_force())
    #     return env.render(800)
    # import moviepy.editor as mpy
    # clip = mpy.VideoClip(make_frame, duration=100)
    # clip.write_videofile("ode.avi", codec="png", fps=20)
    # print('o:{}'.format(env.obj.getPos()))
