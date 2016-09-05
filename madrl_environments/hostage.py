import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding

from rltools.util import EzPickle
from madrl_environments import AbstractMAEnv, Agent


class CircAgent(Agent):

    def __init__(self, idx, radius, n_sensors, sensor_range, addid=True):
        self._idx = idx
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range

        self._sensor_obscoord = 5  # XXX
        self._obscoord_from_sensors = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._obscoord_from_sensors + 5  #+  1  # XXX
        if addid:
            self._obs_dim += 1

        self._position = None
        self._velocity = None
        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self._n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self._sensors = sensor_vecs_K_2

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-10, high=10, shape=(2,))  # x,y

    @property
    def position(self):
        assert self._position is not None
        return self._position

    @property
    def velocity(self):
        assert self._velocity is not None
        return self._velocity

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self._position = x_2

    def set_velocity(self, v_2):
        assert v_2.shape == (2,)
        self._velocity = v_2

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def sensed(self, objx_N_2, same=False):
        """Whether `obj` would be sensed by the pursuers"""
        relpos_obj_N_2 = objx_N_2 - np.expand_dims(self.position, 0)
        sensorvals_K_N = self.sensors.dot(relpos_obj_N_2.T)
        sensorvals_K_N[(sensorvals_K_N < 0) | (sensorvals_K_N > self._sensor_range) | ((
            relpos_obj_N_2**2).sum(axis=1)[None, :] - sensorvals_K_N**2 > self._radius**2)] = np.inf
        if same:
            sensorvals_K_N[:, self._idx - 1] = np.inf
        return sensorvals_K_N


class ContinuousHostageWorld(AbstractMAEnv, EzPickle):

    def __init__(self, n_good, n_hostages, n_bad, n_coop_save, n_coop_avoid, radius=0.015,
                 key_loc=None, bad_speed=0.01, n_sensors=30, sensor_range=0.2, action_scale=0.01,
                 save_reward=5., hit_reward=-1., encounter_reward=0.01, bomb_reward=-5.,
                 bomb_radius=0.05, key_radius=0.0075, control_penalty=-.1, reward_mech='global',
                 addid=True, **kwargs):
        """
        The environment consists of a square world with hostages behind gates. One of the good agent has to find the keys only then the gates can be obtained. Once the gates are opened the good agents need to find the hostages to save them. They also need to avoid the bomb and the bad agents. Coming across a bomb terminates the game and gives a large negative reward
        """
        EzPickle.__init__(self, n_good, n_hostages, n_bad, n_coop_save, n_coop_avoid, radius,
                          key_loc, bad_speed, n_sensors, sensor_range, action_scale, save_reward,
                          hit_reward, encounter_reward, bomb_reward, bomb_radius, key_radius,
                          control_penalty, reward_mech, addid, **kwargs)
        self.n_good = n_good
        self.n_hostages = n_hostages
        self.n_bad = n_bad
        self.n_coop_save = n_coop_save
        self.n_coop_avoid = n_coop_avoid
        self.radius = radius
        self.key_loc = key_loc
        self.key_radius = key_radius
        self.bad_speed = bad_speed
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_good) * sensor_range if isinstance(
            sensor_range, float) else sensor_range
        self.action_scale = action_scale
        self.save_reward = save_reward
        self.hit_reward = hit_reward
        self.encounter_reward = encounter_reward
        self.bomb_reward = bomb_reward
        self.bomb_radius = bomb_radius
        self.control_penalty = control_penalty
        self._reward_mech = reward_mech
        self._addid = addid
        self.seed()

        self._rescuers = [CircAgent(agid + 1, self.radius, self.n_sensors, self.sensor_range[agid],
                                    addid=self._addid) for agid in range(self.n_good)]

        self._criminals = [
            CircAgent(agid + 1, self.radius, self.n_sensors, self.sensor_range.mean())
            for agid in range(self.n_bad)
        ]

        self._hostages = [CircAgent(agid + 1, self.radius * 2, self.n_sensors,
                                    self.sensor_range.min()) for agid in range(self.n_hostages)]

    @property
    def reward_mech(self):
        return self._reward_mech

    @property
    def timestep_limit(self):
        return 1000

    @property
    def agents(self):
        return self._rescuers

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def is_gate_open(self):
        return self._gate_open

    def reset(self):
        self._timesteps = 0
        self._gate_open = False
        self._bombed = False

        # Initialize key location
        if self.key_loc is None:
            self.key_loc = 1 - self.np_random.rand(1, 2) * 0.1
        else:
            assert self.key_loc.ndim == 2

        # Initialize good agents
        # Avoid spawning in the hostage location
        for rescuer in self._rescuers:
            pos = self.np_random.rand(2)
            pos[-1] = np.clip(pos[-1], 0.55, 0.95)
            rescuer.set_position(pos)
            rescuer.set_velocity(np.zeros(2))

        # Initialize hostages
        for hostage in self._hostages:
            pos = self.np_random.rand(2)
            pos[-1] = np.clip(pos[-1], 0, 0.35 + self.np_random.rand() * 0.01)
            hostage.set_position(pos)
            hostage.set_velocity(np.zeros(2))

        self.curr_host_saved_mask = np.zeros(self.n_hostages, dtype=bool)

        # Initialize bad agents
        for criminal in self._criminals:
            pos = self.np_random.rand(2)
            criminal.set_position(pos)
            criminal.set_velocity(self.np_random.rand(2) * self.bad_speed)

        # Bomb location
        self.bomb_loc = np.clip(self.np_random.rand(1, 2), 0., 0.25)

        return self.step(np.zeros((len(self.agents), 2)))[0]

    @property
    def is_terminal(self):
        return self._bombed or self.curr_host_saved_mask.all()

    def _caught(self, is_colliding_N1_N2, n_coop):
        """ Checke whether collision results in catching the object

        This is because you need `n_coop` agents to collide with the object to actually catch it
        """
        # number of N1 colliding with given N2
        n_collisions_N2 = is_colliding_N1_N2.sum(axis=0)
        is_caught_cN2 = np.where(n_collisions_N2 >= n_coop)[0]

        # number of N2 colliding with given N1
        who_collisions_N1_cN2 = is_colliding_N1_N2[:, is_caught_cN2]
        who_caught_cN1 = np.where(who_collisions_N1_cN2 >= 1)[0]

        return is_caught_cN2, who_caught_cN1

    def _closest_dist(self, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        for inp in range(len(sensorvals_Np_K_N)):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]

    def _extract_speed_features(self, objv_N_2, closest_obj_idx_N_K, sensedmask_obj_Np_K):
        sensorvals = []
        for rescuer in self._rescuers:
            sensorvals.append(
                rescuer.sensors.dot((objv_N_2 - np.expand_dims(rescuer.velocity, 0)).T))

        sensed_objspeed_Np_K_N = np.c_[sensorvals]

        sensed_objspeedfeatures_Np_K = np.zeros((self.n_good, self.n_sensors))

        sensorvals = []
        for inp in range(len(self.agents)):
            sensorvals.append(sensed_objspeed_Np_K_N[inp, :, :][np.arange(self.n_sensors),
                                                                closest_obj_idx_N_K[inp, :]])

        sensed_objspeedfeatures_Np_K[sensedmask_obj_Np_K] = np.c_[sensorvals][sensedmask_obj_Np_K]

        return sensed_objspeedfeatures_Np_K

    def step(self, action_Nr2):
        action_Nr2 = np.asarray(action_Nr2)
        action_Nr_2 = action_Nr2.reshape((len(self.agents), 2))
        action_Nr_2 = action_Nr_2 * self.action_scale
        # action_Ng_2 = action_Ng2.reshape((len(self.agents), 2))
        # action_Ng_2 = action_Ng_2 * self.action_scale

        rewards = np.zeros((len(self.agents,)))
        assert action_Nr_2.shape == (len(self.agents), 2)

        for nru, rescuer in enumerate(self._rescuers):
            rescuer.set_velocity(rescuer.velocity + action_Nr_2[nru])
            rescuer.set_position(rescuer.position + rescuer.velocity)

        # self.goodv_Ng_2 += action_Ng_2
        # self.goodx_Ng_2 += self.goodv_Ng_2

        # Penalize large actions
        if self.reward_mech == 'global':
            rewards += self.control_penalty * (action_Nr_2**2).sum()
        else:
            rewards += self.control_penalty * (action_Nr_2**2).sum(axis=1)

        # Players stop on hitting a wall
        for nru, rescuer in enumerate(self._rescuers):
            clippedx_2 = np.clip(rescuer.position, 0, 1)
            vel_2 = rescuer.velocity
            vel_2[rescuer.position != clippedx_2] = 0
            rescuer.set_velocity(vel_2)
            rescuer.set_position(clippedx_2)

        # clippedx_Ng_2 = np.clip(self.goodx_Ng_2, 0, 1)
        # self.goodv_Ng_2[self.goodx_Ng_2 != clippedx_Ng_2] = 0
        # self.goodx_Ng_2 = clippedx_Ng_2

        # Players rebound on hitting a gate
        if not self.is_gate_open:
            for nru, rescuer in enumerate(self._rescuers):
                clippedx_2 = np.clip(rescuer.position, 0.5 + self.radius, 1)
                vel_2 = rescuer.velocity
                vel_2[rescuer.position != clippedx_2] *= -1
                rescuer.set_velocity(vel_2)
                rescuer.set_position(clippedx_2)

            # clippedx_Ng_2 = np.clip(self.goodx_Ng_2, 0.5 + self.radius, 1)
            # self.goodv_Ng_2[self.goodx_Ng_2 != clippedx_Ng_2] *= -1
            # self.goodx_Ng_2 = clippedx_Ng_2

            # Find collisions
        rescuerx_Nr_2 = np.array([rescuer.position for rescuer in self._rescuers])
        criminalx_Nc_2 = np.array([criminal.position for criminal in self._criminals])
        hostagex_Nh_2 = np.array([hostage.position for hostage in self._hostages])

        # Hostage
        hodists_Nr_Nh = ssd.cdist(rescuerx_Nr_2, hostagex_Nh_2)
        is_colliding_ho_Nr_Nh = hodists_Nr_Nh <= np.asarray([rescuer._radius + hostage._radius
                                                             for rescuer in self._rescuers
                                                             for hostage in self._hostages
                                                            ]).reshape(self.n_good, self.n_hostages)
        ho_caught, which_rescuer_caught_ho = self._caught(is_colliding_ho_Nr_Nh, self.n_coop_save)
        ho_encounters, which_rescuer_encounter_ho = self._caught(is_colliding_ho_Nr_Nh, 1)
        # Criminal
        crdists_Nr_Nc = ssd.cdist(rescuerx_Nr_2, criminalx_Nc_2)
        is_colliding_cr_Nr_Nc = crdists_Nr_Nc <= np.asarray([rescuer._radius + criminal._radius
                                                             for rescuer in self._rescuers
                                                             for criminal in self._criminals
                                                            ]).reshape(self.n_good, self.n_bad)
        cr_caught_avoid, which_rescuer_caught_avoid_cr = self._caught(is_colliding_cr_Nr_Nc,
                                                                      self.n_coop_avoid)
        cr_caught, which_rescuer_caught_cr = self._caught(is_colliding_cr_Nr_Nc, 1)

        # Bomb
        bodists_Nr_1 = ssd.cdist(rescuerx_Nr_2, self.bomb_loc)
        is_colliding_bo_Nr_1 = bodists_Nr_1 <= np.asarray(
            [rescuer._radius + self.bomb_radius for rescuer in self._rescuers])
        bo_caught, which_rescuer_caught_bo = self._caught(is_colliding_bo_Nr_1, 1)

        # Key
        kedists_Nr_1 = ssd.cdist(rescuerx_Nr_2, self.key_loc)
        is_colliding_ke_Nr_1 = kedists_Nr_1 <= np.asarray(
            [rescuer._radius + self.key_radius for rescuer in self._rescuers])
        ke_caught, which_rescuer_caught_ke = self._caught(is_colliding_ke_Nr_1, 1)

        # Find sensed objects
        # Hostages
        sensorvals_Nr_K_Nh = np.array([rescuer.sensed(hostagex_Nh_2) for rescuer in self._rescuers])
        sensorvals_Nr_K_Nh[:, :, self.curr_host_saved_mask] = np.inf

        # Criminals
        sensorvals_Nr_K_Nc = np.array(
            [rescuer.sensed(criminalx_Nc_2) for rescuer in self._rescuers])

        # Bomb
        sensorvals_Nr_K_b1 = np.array([rescuer.sensed(self.bomb_loc) for rescuer in self._rescuers])

        # Key
        sensorvals_Nr_K_k1 = np.array([rescuer.sensed(self.key_loc) for rescuer in self._rescuers])

        # Allies
        sensorvals_Nr_K_Nr = np.array(
            [rescuer.sensed(rescuerx_Nr_2, same=True) for rescuer in self._rescuers])

        # dist features
        # Hostages
        closest_ho_idx_Nr_K = np.argmin(sensorvals_Nr_K_Nh, axis=2)
        closest_ho_dist_Nr_K = self._closest_dist(closest_ho_idx_Nr_K, sensorvals_Nr_K_Nh)
        sensedmask_ho_Nr_K = np.isfinite(closest_ho_dist_Nr_K)
        sensed_hodistfeatures_Nr_K = np.zeros((self.n_good, self.n_sensors))
        if self.is_gate_open:
            sensed_hodistfeatures_Nr_K[sensedmask_ho_Nr_K] = closest_ho_dist_Nr_K[
                sensedmask_ho_Nr_K]
        # Criminals
        closest_cr_idx_Nr_K = np.argmin(sensorvals_Nr_K_Nc, axis=2)
        closest_cr_dist_Nr_K = self._closest_dist(closest_cr_idx_Nr_K, sensorvals_Nr_K_Nc)
        sensedmask_cr_Nr_K = np.isfinite(closest_cr_dist_Nr_K)
        sensed_crdistfeatures_Nr_K = np.zeros((self.n_good, self.n_sensors))
        sensed_crdistfeatures_Nr_K[sensedmask_cr_Nr_K] = closest_cr_dist_Nr_K[sensedmask_cr_Nr_K]
        # Bomb
        closest_bo_idx_Nr_K = np.argmin(sensorvals_Nr_K_b1, axis=2)
        closest_bo_dist_Nr_K = self._closest_dist(closest_bo_idx_Nr_K, sensorvals_Nr_K_b1)
        sensedmask_bo_Nr_K = np.isfinite(closest_bo_dist_Nr_K)
        sensed_bodistfeatures_Nr_K = np.zeros((self.n_good, self.n_sensors))
        sensed_bodistfeatures_Nr_K[sensedmask_bo_Nr_K] = closest_bo_dist_Nr_K[sensedmask_bo_Nr_K]
        # Key
        closest_ke_idx_Nr_K = np.argmin(sensorvals_Nr_K_k1, axis=2)
        closest_ke_dist_Nr_K = self._closest_dist(closest_ke_idx_Nr_K, sensorvals_Nr_K_k1)
        sensedmask_ke_Nr_K = np.isfinite(closest_ke_dist_Nr_K)
        sensed_kedistfeatures_Nr_K = np.zeros((self.n_good, self.n_sensors))
        if not self.is_gate_open:
            sensed_kedistfeatures_Nr_K[sensedmask_ke_Nr_K] = closest_ke_dist_Nr_K[
                sensedmask_ke_Nr_K]
        # Allies
        closest_re_idx_Nr_K = np.argmin(sensorvals_Nr_K_Nr, axis=2)
        closest_re_dist_Nr_K = self._closest_dist(closest_re_idx_Nr_K, sensorvals_Nr_K_Nr)
        sensedmask_re_Nr_K = np.isfinite(closest_re_dist_Nr_K)
        sensed_redistfeatures_Nr_K = np.zeros((self.n_good, self.n_sensors))
        sensed_redistfeatures_Nr_K[sensedmask_re_Nr_K] = closest_re_dist_Nr_K[sensedmask_re_Nr_K]

        # speed features
        rescuerv_Nr_2 = np.array([rescuer.velocity for rescuer in self._rescuers])
        criminalv_Nc_2 = np.array([criminal.velocity for criminal in self._criminals])
        hostagev_Nh_2 = np.array([hostage.velocity for hostage in self._hostages])

        # Criminals
        sensed_crspeedfeatures_Nr_K = self._extract_speed_features(criminalv_Nc_2,
                                                                   closest_cr_idx_Nr_K,
                                                                   sensedmask_cr_Nr_K)
        # Allies
        sensed_respeedfeatures_Nr_K = self._extract_speed_features(rescuerv_Nr_2,
                                                                   closest_re_idx_Nr_K,
                                                                   sensedmask_re_Nr_K)

        # Process collisions
        if ho_caught.size:
            for hoc in ho_caught:
                self.curr_host_saved_mask[hoc] = True

        if cr_caught.size:
            for crc in cr_caught:
                self._criminals[crc].set_position(self.np_random.rand(2))
                self._criminals[crc].set_velocity((self.np_random.rand(2) - 0.5) * self.bad_speed)

        if bo_caught.size:
            self._bombed = True

        if ke_caught.size:
            self._gate_open = True

        if self.reward_mech == 'global':
            rewards += (len(ho_encounters) * self.encounter_reward * self._gate_open +
                        len(ho_caught) * self.save_reward + len(cr_caught) *
                        self.hit_reward  # - ba_catches * self.hit_reward
                        + self._bombed * self.bomb_reward)
        else:
            rewards[which_rescuer_caught_ho] += self.save_reward
            rewards[which_rescuer_encounter_ho] += self.encounter_reward * self._gate_open
            rewards[which_rescuer_caught_cr] += self.hit_reward
            rewards[which_rescuer_caught_bo] += self._bombed * self.bomb_reward

        # Add features together
        sensorfeatures_Nr_K_O = np.c_[sensed_crdistfeatures_Nr_K, sensed_crspeedfeatures_Nr_K,
                                      sensed_hodistfeatures_Nr_K, sensed_kedistfeatures_Nr_K,
                                      sensed_bodistfeatures_Nr_K]

        for criminal in self._criminals:
            # Everybody move
            criminal.set_position(criminal.position + criminal.velocity)
            # Bounce object if it hits a wall
            if all(criminal.position != np.clip(criminal.position, 0, 1)):
                criminal.set_velocity(-1 * criminal.velocity)

        obslist = []
        for inp in range(len(self.agents)):
            if self._addid:
                obslist.append(
                    np.concatenate([sensorfeatures_Nr_K_O[inp, ...].ravel(), [float((
                        is_colliding_ho_Nr_Nh[inp, :]).sum() > 0), float((is_colliding_cr_Nr_Nc[
                            inp, :]).sum() > 0), float((is_colliding_ke_Nr_1[inp, :]).sum(
                            ) > 0), float((is_colliding_bo_Nr_1[inp, :]).sum() > 0)], [float(
                                self.is_gate_open)], [inp + 1]]))
            else:
                obslist.append(
                    np.concatenate([sensorfeatures_Nr_K_O[inp, ...].ravel(), [float((
                        is_colliding_ho_Nr_Nh[inp, :]).sum() > 0), float((is_colliding_cr_Nr_Nc[
                            inp, :]).sum() > 0), float((is_colliding_ke_Nr_1[inp, :]).sum(
                            ) > 0), float((is_colliding_bo_Nr_1[inp, :]).sum() > 0)], [float(
                                self.is_gate_open)]]))

        self._timesteps += 1
        done = self.is_terminal
        info = dict(ho_saved=len(ho_caught), cr_encs=len(cr_caught))

        return obslist, rewards, done, info

    def render(self, screen_size=800, rate=10):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255

        # Stationary objects
        if not self.is_gate_open:
            color = (128, 128, 200)
            cv2.line(img, tuple(np.array([0 * screen_size, 0.5 * screen_size]).astype(int)),
                     tuple(np.array([1 * screen_size, 0.5 * screen_size]).astype(int)), color, 1,
                     lineType=cv2.CV_AA)

            keyx_2 = np.squeeze(self.key_loc)
            assert keyx_2.shape == (2,)
            kcolor = (0, 0, 255)
            cv2.circle(img, tuple((keyx_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), kcolor, -1, lineType=cv2.CV_AA)

        for ibomb, bombx_2 in enumerate(self.bomb_loc):
            assert bombx_2.shape == (2,)
            color = (0, 0, 255)
            cv2.circle(img, tuple((bombx_2 * screen_size).astype(int)),
                       int(self.bomb_radius * screen_size), color, -1, lineType=cv2.CV_AA)

        for hostage in np.asarray(self._hostages)[~self.curr_host_saved_mask]:
            color = (255, 120, 0)
            cv2.circle(img, tuple((hostage.position * screen_size).astype(int)),
                       int(hostage._radius * screen_size), color, -1, lineType=cv2.CV_AA)

        for criminal in self._criminals:
            color = (50, 50, 200)
            cv2.circle(img, tuple((criminal.position * screen_size).astype(int)),
                       int(criminal._radius * screen_size), color, -1, lineType=cv2.CV_AA)

        for rescuer in self._rescuers:
            for k in range(rescuer._n_sensors):
                color = (0, 0, 0)
                cv2.line(img, tuple((rescuer.position * screen_size).astype(int)),
                         tuple(((rescuer.position + rescuer._sensor_range * rescuer.sensors[k]) *
                                screen_size).astype(int)), color, 1, lineType=cv2.CV_AA)
                cv2.circle(img, tuple((rescuer.position * screen_size).astype(int)),
                           int(rescuer._radius * screen_size), (255, 0, 0), -1, lineType=cv2.CV_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Hostage', img)
        cv2.waitKey(rate)


if __name__ == '__main__':
    env = ContinuousHostageWorld(3, 10, 5, 2, 2)
    obs = env.reset()
    while True:
        obs, rew, done, _ = env.step(env.np_random.randn(6) * .5)
        if rew.sum() > 0:
            print(rew)
        env.render()
        if done:
            break
