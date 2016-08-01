import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces


class ContinuousHostageWorld(object):

    def __init__(self, n_good, n_hostages, n_bad, n_coop_save, n_coop_avoid, radius=0.015,
                 key_loc=np.array([1, 1]), bad_speed=0.01, n_sensors=30, sensor_range=0.2,
                 action_scale=0.01, save_reward=5., hit_reward=-1., encounter_reward=0.01,
                 bomb_reward=-5., bomb_radius=0.03, control_penalty=-.5, centralized=True,
                 **kwargs):
        """
        The environment consists of a square world with hostages behind gates. One of the good agent has to find the keys only then the gates can be obtained. Once the gates are opened the good agents need to find the hostages to save them. They also need to avoid the bomb and the bad agents. Coming across a bomb terminates the game and gives a large negative reward
        """
        self.n_good = n_good
        self.n_hostages = n_hostages
        self.n_bad = n_bad
        self.n_coop_save = n_coop_save
        self.n_coop_avoid = n_coop_avoid
        self.radius = radius
        self.key_loc = key_loc
        self.bad_speed = bad_speed
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range
        self.action_scale = action_scale
        self.save_reward = save_reward
        self.hit_reward = hit_reward
        self.encounter_reward = encounter_reward
        self.bomb_reward = bomb_reward
        self.bomb_radius = bomb_radius
        self.control_penanlty = control_penalty
        self.centralized = centralized

    # Observations
    # Distance from other agents (sensor_range)
    # Speed of other agents (sensor_range)
    # Whether key
    # Own id

    @property
    def is_gate_open(self):
        return self._gate_open

    def reset(self):
        self._gate_open = False
        self._bombed = False

        # Initialize key location
        self._key_loc = 0.5 - np.random.rand(1, 2) * 0.5

        # Initialize good agents
        # Avoid spawning in the hostage location
        self.goodx_Ng_2 = 1 - np.random.rand(self.n_good, 2) * 0.5
        self.goodv_Ng_2 = np.zeros((self.n_good, 2))

        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self.n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self.sensor_vecs_Ng_K_2 = np.tile(sensor_vecs_K_2, (self.n_good, 1, 1))

        # Initialize hostages
        #
        self.hostagex_Nh_2 = np.random.rand(self.n_hostages, 2) * np.array([1, 0.5])
        self.hostagev_Nh_2 = np.zeros((self.n_hostages, 2))

        self.curr_host_saved_mask = np.zeros(self.n_hostages, dtype=bool)

        # Initialize bad agents
        self.badx_Nb_2 = np.random.rand(self.n_bad, 2) * 0.5
        self.badv_Nb_2 = (np.random.rand(self.n_bad, 2) - 0.5) * self.bad_speed

        # Bomb location
        self.bomb_loc = np.random.rand(1, 2) * 0.25

        return self.step(np.zeros((self.n_good, 2)))[0]

    @property
    def is_terminal(self):
        return self._bombed

    def _caught(self, is_colliding_Np_Ne, n_coop):
        """ Checke whether collision results in catching the object

        This is because you need `n_coop` agents to collide with the object to actually catch it
        """
        n_collisions_Ne = is_colliding_Np_Ne.sum(axis=0)
        is_caught_Ne = n_collisions_Ne >= n_coop
        catches = is_caught_Ne.sum()
        return catches, is_caught_Ne

    def _sensed(self, objx_N_2):
        """Whether `obj` would be sensed by the pursuers"""
        relpos_obj_N_Ng_2 = objx_N_2[:, None, :] - self.goodx_Ng_2
        sensorvals_Ng_K_N = np.tensordot(self.sensor_vecs_Ng_K_2, relpos_obj_N_Ng_2.transpose(1, 0,
                                                                                              2),
                                         axes=(2, 2))[0].transpose(1, 0, 2)
        sensorvals_Ng_K_N[(sensorvals_Ng_K_N < 0) | (sensorvals_Ng_K_N > self.sensor_range) | (
            (relpos_obj_N_Ng_2**2).sum(axis=2).T[:, None, ...] - sensorvals_Ng_K_N**2 > self.radius
            **2)] = np.inf
        return sensorvals_Ng_K_N

    def _closest_dist(self, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        for inp in range(self.n_good):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]

    def _extract_speed_features(self, objv_N_2, closest_obj_idx_N_K, sensedmask_obj_Ng_K):
        assert closest_obj_idx_N_K.shape == sensedmask_obj_Ng_K.shape

        sensed_objspeed_Ng_K_N = np.tensordot(self.sensor_vecs_Ng_K_2, (
            objv_N_2[:, None, ...] - self.goodv_Ng_2).transpose(1, 0, 2),
                                              axes=(2, 2))[0].transpose(1, 0, 2)
        assert sensed_objspeed_Ng_K_N.shape[-1] >= closest_obj_idx_N_K.max()
        sensed_objspeedfeatures_Ng_K = np.zeros((self.n_good, self.n_sensors))

        sensorvals = []
        for inp in range(self.n_good):
            tmp = sensed_objspeed_Ng_K_N[inp, :, :][np.arange(self.n_sensors), closest_obj_idx_N_K[
                inp, :]]
            sensorvals.append(tmp)
        sensed_objspeedfeatures_Ng_K[sensedmask_obj_Ng_K] = np.c_[sensorvals][sensedmask_obj_Ng_K]

        return sensed_objspeedfeatures_Ng_K

    def step(self, action_Ng2):
        action_Ng_2 = action_Ng2.reshape(self.n_good, 2)
        action_Ng_2 = action_Ng_2 * self.action_scale

        assert action_Ng_2.shape == (self.n_good, 2)

        reward = 0.

        self.goodv_Ng_2 += action_Ng_2
        self.goodx_Ng_2 += self.goodv_Ng_2

        reward += self.control_penanlty * (action_Ng_2**2).sum()

        # Players stop on hitting a wall
        clippedx_Ng_2 = np.clip(self.goodx_Ng_2, 0, 1)
        self.goodv_Ng_2[self.goodx_Ng_2 != clippedx_Ng_2] = 0
        self.goodx_Ng_2 = clippedx_Ng_2

        # Players rebound on hitting a wall
        if not self.is_gate_open:
            clippedx_Ng_2 = np.clip(self.goodx_Ng_2, 0.5 + self.radius, 1)
            self.goodv_Ng_2[self.goodx_Ng_2 != clippedx_Ng_2] *= -1
            self.goodx_Ng_2 = clippedx_Ng_2

        # Find collisions
        # Hostages
        hodists_Ng_Nh = ssd.cdist(self.goodx_Ng_2, self.hostagex_Nh_2)
        is_colliding_ho_Ng_Nh = hodists_Ng_Nh <= self.radius * 2
        # num_collisions depends on how many needed to save a hostage
        ho_catches, ho_caught_Nh = self._caught(is_colliding_ho_Ng_Nh, self.n_coop_save)

        # Bad
        badists_Ng_Nb = ssd.cdist(self.goodx_Ng_2, self.badx_Nb_2)
        is_colliding_ba_Ng_Nh = badists_Ng_Nb <= self.radius * 2
        # Find the alone caught ones and the coop ones
        # penalize (hit) if good caught alone otherwise nothing
        ba_catches_alone, ba_caught_alone_Nb = self._caught(is_colliding_ba_Ng_Nh, 1)
        ba_catches, ba_caught_Nb = self._caught(is_colliding_ba_Ng_Nh, self.n_coop_avoid)

        # Key
        kedists_Ng_1 = ssd.cdist(self.goodx_Ng_2, self.key_loc[None, :])
        is_colliding_ke_Ng_1 = kedists_Ng_1 <= self.radius * 2
        ke_catches, ke_caught_1 = self._caught(is_colliding_ke_Ng_1, 1)

        # Bomb
        bodists_Ng_1 = ssd.cdist(self.goodx_Ng_2, self.bomb_loc)
        is_colliding_bo_Ng_1 = bodists_Ng_1 <= self.radius + self.bomb_radius
        bo_catches, bo_caught_1 = self._caught(is_colliding_bo_Ng_1, 1)

        # Find sensed
        # Hostages
        sensorvals_Ng_K_Nh = self._sensed(self.hostagex_Nh_2)
        sensorvals_Ng_K_Nh[:, :, self.curr_host_saved_mask] = np.inf

        # Bad
        sensorvals_Ng_K_Nb = self._sensed(self.badx_Nb_2)

        # Key
        sensorvals_Ng_K_ke1 = self._sensed(self.key_loc[None, :])

        # Bomb
        sensorvals_Ng_K_bo1 = self._sensed(self.bomb_loc)

        # dist features
        closest_ho_idx_Ng_K = np.argmin(sensorvals_Ng_K_Nh, axis=2)
        closest_ho_dist_Ng_K = self._closest_dist(closest_ho_idx_Ng_K, sensorvals_Ng_K_Nh)
        sensedmask_ho_Ng_K = np.isfinite(closest_ho_dist_Ng_K)
        sensed_hodistfeatures_Ng_K = np.zeros((self.n_good, self.n_sensors))
        sensed_hodistfeatures_Ng_K[sensedmask_ho_Ng_K] = closest_ho_dist_Ng_K[sensedmask_ho_Ng_K]

        closest_ba_idx_Ng_K = np.argmin(sensorvals_Ng_K_Nb, axis=2)
        closest_ba_dist_Ng_K = self._closest_dist(closest_ba_idx_Ng_K, sensorvals_Ng_K_Nb)
        sensedmask_ba_Ng_K = np.isfinite(closest_ba_dist_Ng_K)
        sensed_badistfeatures_Ng_K = np.zeros((self.n_good, self.n_sensors))
        sensed_badistfeatures_Ng_K[sensedmask_ba_Ng_K] = closest_ba_dist_Ng_K[sensedmask_ba_Ng_K]

        closest_ke_idx_Ng_K = np.argmin(sensorvals_Ng_K_ke1, axis=2)
        closest_ke_dist_Ng_K = self._closest_dist(closest_ke_idx_Ng_K, sensorvals_Ng_K_ke1)
        sensedmask_ke_Ng_K = np.isfinite(closest_ke_dist_Ng_K)
        sensed_kedistfeatures_Ng_K = np.zeros((self.n_good, self.n_sensors))
        sensed_kedistfeatures_Ng_K[sensedmask_ke_Ng_K] = closest_ke_dist_Ng_K[sensedmask_ke_Ng_K]

        closest_bo_idx_Ng_K = np.argmin(sensorvals_Ng_K_bo1, axis=2)
        closest_bo_dist_Ng_K = self._closest_dist(closest_bo_idx_Ng_K, sensorvals_Ng_K_bo1)
        sensedmask_bo_Ng_K = np.isfinite(closest_bo_dist_Ng_K)
        sensed_bodistfeatures_Ng_K = np.zeros((self.n_good, self.n_sensors))
        sensed_bodistfeatures_Ng_K[sensedmask_bo_Ng_K] = closest_bo_dist_Ng_K[sensedmask_bo_Ng_K]

        # speed features
        # Bad

        sensed_baspeedfeatures_Ng_K = self._extract_speed_features(self.badv_Nb_2,
                                                                   closest_ba_idx_Ng_K,
                                                                   sensedmask_ba_Ng_K)

        # Process collisions
        self.curr_host_saved_mask[ho_caught_Nh] = np.ones(ho_catches, dtype=bool)

        self.badx_Nb_2[ba_caught_Nb, :] = np.random.rand(ba_catches, 2) * 0.5

        if bo_catches > 0:
            self._bombed = True

        if ke_catches > 0:
            self._gate_open = True

        reward += ho_catches * self.save_reward + ba_catches_alone * self.hit_reward - ba_catches * self.hit_reward + self._bombed * self.bomb_reward

        # Add features together
        sensorfeatures_Ng_K_O = np.c_[sensed_badistfeatures_Ng_K, sensed_baspeedfeatures_Ng_K,
                                      sensed_hodistfeatures_Ng_K, sensed_kedistfeatures_Ng_K,
                                      sensed_bodistfeatures_Ng_K]

        self.badx_Nb_2 += self.badv_Nb_2

        # Bounce object if it hits a wall
        self.badv_Nb_2[np.clip(self.badx_Nb_2, 0, 1) != self.badx_Nb_2] *= -1

        obslist = []
        for inp in range(self.n_good):
            obslist.append(
                np.concatenate([sensorfeatures_Ng_K_O[inp, ...].ravel(
                ), [float((is_colliding_ho_Ng_Nh[inp, :]).sum() > 0), float((is_colliding_ba_Ng_Nh[
                    inp, :]).sum() > 0), float((is_colliding_ke_Ng_1[inp, :]).sum() > 0), float((
                        is_colliding_bo_Ng_1[inp, :]).sum() > 0)], [inp + 1]]))

        if self.centralized:
            obs = np.c_[obslist].ravel()

        else:
            obs = obslist

        done = self.is_terminal
        info = None

        return obs, reward, done, info

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
        for ibomb, bombx_2 in enumerate(self.bomb_loc):
            assert bombx_2.shape == (2,)
            color = (0, 0, 255)
            cv2.circle(img, tuple((bombx_2 * screen_size).astype(int)),
                       int(self.bomb_radius * screen_size), color, -1, lineType=cv2.CV_AA)

        keyx_2 = self.key_loc
        assert keyx_2.shape == (2,)
        color = (0, 0, 255)
        cv2.circle(img, tuple((keyx_2 * screen_size).astype(int)), int(self.radius * screen_size),
                   color, -1, lineType=cv2.CV_AA)

        for ihost, hostagex_2 in enumerate(self.hostagex_Nh_2[np.invert(
                self.curr_host_saved_mask)]):
            assert hostagex_2.shape == (2,)
            color = (255, 0, 0)
            cv2.circle(img, tuple((hostagex_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), color, -1, lineType=cv2.CV_AA)

        for ibad, badx_2 in enumerate(self.badx_Nb_2):
            assert badx_2.shape == (2,)
            color = (50, 50, 200)
            cv2.circle(img, tuple((badx_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), color, -1, lineType=cv2.CV_AA)

        for igood, goodx_2 in enumerate(self.goodx_Ng_2):
            assert goodx_2.shape == (2,)

            for k in range(self.n_sensors):
                color = (0, 0, 0)
                cv2.line(img, tuple((goodx_2 * screen_size).astype(int)),
                         tuple(((goodx_2 + self.sensor_range * self.sensor_vecs_Ng_K_2[igood, k, :])
                                * screen_size).astype(int)), color, 1, lineType=cv2.CV_AA)

            cv2.circle(img, tuple((goodx_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), (0, 255, 0), -1, lineType=cv2.CV_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Hostage', img)
        cv2.waitKey(rate)

    def animate(self, act_fn, nsteps, file_name, rate=20):
        o = self.reset()
        self.render(rate=rate)
        rew = 0
        for i in range(nsteps):
            if self.centralized:
                a, adist = act_fn(o)
            else:
                a = []
                for i, agent_o in enumerate(o):
                    agent_a, adist = act_fn(agent_o)
                    a.append(agent_a)
                    a = np.asarray(a)

            o, r, done, _ = self.step(a)
            rew += r
            if r > 0:
                print(r)
                self.render(rate=rate)
            if done:
                break
        return rew


if __name__ == '__main__':
    env = ContinuousHostageWorld(3, 10, 5, 2, 2)
    obs = env.reset()
    while True:
        obs, rew, done, _ = env.step(np.random.randn(6) * .5)
        if rew > 0:
            print(rew)
        env.render()
        if done:
            break
