import numpy as np
import scipy.spatial.distance as ssd
from gym import spaces
from gym.utils import seeding

from madrl_environments import AbstractMAEnv, Agent
from rltools.util import EzPickle


class Archea(Agent):

    def __init__(self, radius, n_sensors, sensor_range):
        self._radius = radius
        self._n_sensors = n_sensors
        self._sensor_range = sensor_range
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 7
        self._obscoord_from_sensors = self._n_sensors * self._sensor_obscoord
        self._obs_dim = self._obscoord_from_sensors + 2 + 1  #2 for type, 1 for id

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,))


class MAWaterWorld(AbstractMAEnv, EzPickle):

    def __init__(self, n_pursuers, n_evaders, n_coop=2, n_poison=10, radius=0.015,
                 obstacle_radius=0.2, obstacle_loc=np.array([0.5, 0.5]), ev_speed=0.01,
                 poison_speed=0.01, n_sensors=30, sensor_range=0.2, action_scale=0.01,
                 poison_reward=-1., food_reward=1., encounter_reward=.05, control_penalty=-.5,
                 reward_mech='global', **kwargs):
        EzPickle.__init__(self, n_pursuers, n_evaders, n_coop, n_poison, radius, obstacle_radius,
                          obstacle_loc, ev_speed, poison_speed, n_sensors, sensor_range,
                          action_scale, poison_reward, food_reward, encounter_reward,
                          control_penalty, reward_mech, **kwargs)
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_coop = n_coop
        self.n_poison = n_poison
        self.obstacle_radius = obstacle_radius
        self.obstacle_loc = obstacle_loc
        self.poison_speed = poison_speed
        self.radius = radius
        self.ev_speed = ev_speed
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_pursuers) * sensor_range
        self.action_scale = action_scale
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.control_penalty = control_penalty
        self.encounter_reward = encounter_reward

        self.n_obstacles = 1
        # So the way it works is that you have the waterworld environment
        # In the centralized setting all observations from each agent are joined together in a single 1D array
        # However, in the decentralized setting we get a list of actions for each agent
        # and output observations for each agent
        # TODO: not sure if the observation shape should include the number of agent.
        # IMHO not
        self._reward_mech = reward_mech
        self.seed()

    @property
    def reward_mech(self):
        return self._reward_mech

    @property
    def timestep_limit(self):
        return 1000

    @property
    def agents(self):
        return [Archea(self.radius, self.n_sensors, self.sensor_range)
                for _ in range(self.n_pursuers)]

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _respawn(self, objx_N_2):
        for i in range(len(objx_N_2)):
            while ssd.cdist(objx_N_2[i, None, :],
                            self.obstaclesx_No_2) <= self.radius + self.obstacle_radius:
                objx_N_2[i, :] = self.np_random.rand(2)
        return objx_N_2

    def reset(self):
        # Initialize obstacles
        if self.obstacle_loc is None:
            self.obstaclesx_No_2 = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstaclesx_No_2 = self.obstacle_loc[None, :]
        self.obstaclesv_No_2 = np.zeros((self.n_obstacles, 2))

        # Initialize pursuers
        self.pursuersx_Np_2 = self.np_random.rand(self.n_pursuers, 2)
        # Avoid spawning where the obstacles lie
        self.pursuersx_Np_2 = self._respawn(self.pursuersx_Np_2)
        self.pursuersv_Np_2 = np.zeros((self.n_pursuers, 2))

        # Sensors
        angles_K = np.linspace(0., 2. * np.pi, self.n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self.sensor_vecs_Np_K_2 = np.tile(sensor_vecs_K_2, (self.n_pursuers, 1, 1))

        # Initialize evaders
        self.evadersx_Ne_2 = self.np_random.rand(self.n_evaders, 2)
        self.evadersx_Ne_2 = self._respawn(self.evadersx_Ne_2)
        self.evadersv_Ne_2 = (self.np_random.rand(self.n_evaders, 2) - .5
                             ) * self.ev_speed  # Random speeds TODO policy?

        # Initialize poisons
        self.poisonx_Npo_2 = self.np_random.rand(self.n_poison, 2)
        self.poisonx_Npo_2 = self._respawn(self.poisonx_Npo_2)
        self.poisonv_Npo_2 = (
            self.np_random.rand(self.n_poison, 2) - .5) * self.poison_speed  # Random speeds

        return self.step(np.zeros((self.n_pursuers, 2)))[0]

    @property
    def is_terminal(self):
        return False

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
        relpos_obj_N_Np_2 = objx_N_2[:, None, :] - self.pursuersx_Np_2
        # sensorvals = []
        # for inp in range(self.n_pursuers):
        #     sensorvals.append(self.sensor_vecs_Np_K_2[inp, ...].dot(relpos_obj_N_Np_2[:, inp, :].T))

        # sensorvals_Np_K_N = np.c_[sensorvals]
        sensorvals_Np_K_N = np.tensordot(self.sensor_vecs_Np_K_2, relpos_obj_N_Np_2.transpose(1, 0,
                                                                                              2),
                                         axes=(2, 2))[0].transpose(1, 0, 2)

        sensorvals_Np_K_N[(sensorvals_Np_K_N < 0) | (
            sensorvals_Np_K_N > self.sensor_range[:, None, None]) | ((relpos_obj_N_Np_2**2).sum(
                axis=2).T[:, None, ...] - sensorvals_Np_K_N**2 > self.radius**2)] = np.inf
        return sensorvals_Np_K_N

    def _closest_dist(self, closest_obj_idx_Np_K, sensorvals_Np_K_N):
        """Closest distances according to `idx`"""
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(sensorvals_Np_K_N[inp, ...][np.arange(self.n_sensors),
                                                          closest_obj_idx_Np_K[inp, ...]])
        return np.c_[sensorvals]

    def _extract_speed_features(self, objv_N_2, closest_obj_idx_N_K, sensedmask_obj_Np_K):
        # sensorvals = []
        # for inp in range(self.n_pursuers):
        #     sensorvals.append(self.sensor_vecs_Np_K_2[inp, ...].dot((objv_N_2 - self.pursuersv_Np_2[
        #         inp, ...]).T))
        # sensed_objspeed_Np_K_N = np.c_[sensorvals]
        sensed_objspeed_Np_K_N = np.tensordot(self.sensor_vecs_Np_K_2, (
            objv_N_2[:, None, ...] - self.pursuersv_Np_2).transpose(1, 0, 2),
                                              axes=(2, 2))[0].transpose(1, 0, 2)
        sensed_objspeedfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))

        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(sensed_objspeed_Np_K_N[inp, :, :][np.arange(self.n_sensors),
                                                                closest_obj_idx_N_K[inp, :]])
        sensed_objspeedfeatures_Np_K[sensedmask_obj_Np_K] = np.c_[sensorvals][sensedmask_obj_Np_K]

        return sensed_objspeedfeatures_Np_K

    def step(self, action_Np2):
        action_Np_2 = action_Np2.reshape(self.n_pursuers, 2)
        # Players
        actions_Np_2 = action_Np_2 * self.action_scale

        rewards = np.zeros((self.n_pursuers,))
        assert action_Np_2.shape == (self.n_pursuers, 2)

        self.pursuersv_Np_2 += actions_Np_2
        self.pursuersx_Np_2 += self.pursuersv_Np_2

        # Penalize large actions
        if self.reward_mech == 'global':
            rewards += self.control_penalty * (actions_Np_2**2).sum()
        else:
            rewards += self.control_penalty * (actions_Np_2**2).sum(axis=1)

        # Players stop on hitting a wall
        clippedx_Np_2 = np.clip(self.pursuersx_Np_2, 0, 1)
        self.pursuersv_Np_2[self.pursuersx_Np_2 != clippedx_Np_2] = 0
        self.pursuersx_Np_2 = clippedx_Np_2

        # Particles rebound on hitting an obstacle
        obsdists_Np_No = ssd.cdist(self.pursuersx_Np_2, self.obstaclesx_No_2)
        is_colliding_obs_Np_No = obsdists_Np_No <= self.radius + self.obstacle_radius
        num_obs_collisions = is_colliding_obs_Np_No.sum()

        is_colliding_obs_Np = is_colliding_obs_Np_No.any(axis=1)
        self.pursuersv_Np_2[is_colliding_obs_Np] *= -1

        obsdists_Ne_No = ssd.cdist(self.evadersx_Ne_2, self.obstaclesx_No_2)
        is_colliding_obs_Ne_No = obsdists_Ne_No <= self.radius + self.obstacle_radius

        is_colliding_obs_Ne = is_colliding_obs_Ne_No.any(axis=1)
        self.evadersv_Ne_2[is_colliding_obs_Ne] *= -1

        obsdists_Npo_No = ssd.cdist(self.poisonx_Npo_2, self.obstaclesx_No_2)
        is_colliding_obs_Npo_No = obsdists_Npo_No <= self.radius + self.obstacle_radius

        is_colliding_obs_Npo = is_colliding_obs_Npo_No.any(axis=1)
        self.poisonv_Npo_2[is_colliding_obs_Npo] *= -1

        # Find collisions
        # Evaders
        evdists_Np_Ne = ssd.cdist(self.pursuersx_Np_2, self.evadersx_Ne_2)
        is_colliding_ev_Np_Ne = evdists_Np_Ne <= self.radius * 2
        # num_collisions depends on how many needed to catch an evader
        ev_catches, ev_caught_Ne = self._caught(is_colliding_ev_Np_Ne, self.n_coop)

        # Poisons
        podists_Np_Npo = ssd.cdist(self.pursuersx_Np_2, self.poisonx_Npo_2)
        is_colliding_po_Np_Npo = podists_Np_Npo <= self.radius * 2
        num_poison_collisions = is_colliding_po_Np_Npo.sum()

        # Find sensed objects
        # Obstacles
        sensorvals_Np_K_No = self._sensed(self.obstaclesx_No_2)

        # Evaders
        sensorvals_Np_K_Ne = self._sensed(self.evadersx_Ne_2)

        # Poison
        sensorvals_Np_K_Npo = self._sensed(self.poisonx_Npo_2)

        # Allies
        sensorvals_Np_K_Np = self._sensed(self.pursuersx_Np_2)

        # dist features
        closest_ob_idx_Np_K = np.argmin(sensorvals_Np_K_No, axis=2)
        closest_ob_dist_Np_K = self._closest_dist(closest_ob_idx_Np_K, sensorvals_Np_K_No)
        sensedmask_ob_Np_K = np.isfinite(closest_ob_dist_Np_K)
        sensed_obdistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_obdistfeatures_Np_K[sensedmask_ob_Np_K] = closest_ob_dist_Np_K[sensedmask_ob_Np_K]
        # Evaders
        closest_ev_idx_Np_K = np.argmin(sensorvals_Np_K_Ne, axis=2)
        closest_ev_dist_Np_K = self._closest_dist(closest_ev_idx_Np_K, sensorvals_Np_K_Ne)
        sensedmask_ev_Np_K = np.isfinite(closest_ev_dist_Np_K)
        sensed_evdistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_evdistfeatures_Np_K[sensedmask_ev_Np_K] = closest_ev_dist_Np_K[sensedmask_ev_Np_K]
        # Poison
        closest_po_idx_Np_K = np.argmin(sensorvals_Np_K_Npo, axis=2)
        closest_po_dist_Np_K = self._closest_dist(closest_po_idx_Np_K, sensorvals_Np_K_Npo)
        sensedmask_po_Np_K = np.isfinite(closest_po_dist_Np_K)
        sensed_podistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_podistfeatures_Np_K[sensedmask_po_Np_K] = closest_po_dist_Np_K[sensedmask_po_Np_K]
        # Allies
        closest_pu_idx_Np_K = sensorvals_Np_K_Np.argsort(axis=2)[..., 1]
        closest_pu_dist_Np_K = self._closest_dist(closest_pu_idx_Np_K, sensorvals_Np_K_Np)
        sensedmask_pu_Np_K = np.isfinite(closest_pu_dist_Np_K)
        sensed_pudistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_pudistfeatures_Np_K[sensedmask_pu_Np_K] = closest_pu_dist_Np_K[sensedmask_pu_Np_K]

        # speed features
        # Evaders
        sensed_evspeedfeatures_Np_K = self._extract_speed_features(self.evadersv_Ne_2,
                                                                   closest_ev_idx_Np_K,
                                                                   sensedmask_ev_Np_K)
        # Poison
        sensed_pospeedfeatures_Np_K = self._extract_speed_features(self.poisonv_Npo_2,
                                                                   closest_po_idx_Np_K,
                                                                   sensedmask_po_Np_K)
        # Allies
        sensed_puspeedfeatures_Np_K = self._extract_speed_features(self.pursuersv_Np_2,
                                                                   closest_pu_idx_Np_K,
                                                                   sensedmask_pu_Np_K)

        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        self.evadersx_Ne_2[ev_caught_Ne, :] = self.np_random.rand(ev_catches, 2)
        self.evadersx_Ne_2[ev_caught_Ne, :] = self._respawn(self.evadersx_Ne_2[ev_caught_Ne, :])
        self.evadersv_Ne_2[ev_caught_Ne, :] = (
            self.np_random.rand(ev_catches, 2) - .5) * self.ev_speed

        po_catches, po_caught_Npo = self._caught(is_colliding_po_Np_Npo, 1)
        self.poisonx_Npo_2[po_caught_Npo, :] = self.np_random.rand(po_catches, 2)
        self.poisonx_Npo_2[po_caught_Npo, :] = self._respawn(self.poisonx_Npo_2[po_caught_Npo, :])
        self.poisonv_Npo_2[po_caught_Npo, :] = (
            self.np_random.rand(po_catches, 2) - .5) * self.poison_speed

        ev_encounters, _ = self._caught(is_colliding_ev_Np_Ne, 1)
        # Update reward based on these collisions
        if self.reward_mech == 'global':
            rewards += ev_catches * self.food_reward + po_catches * self.poison_reward + ev_encounters * self.encounter_reward
        else:
            raise NotImplementedError()

        # Add features together
        sensorfeatures_Np_K_O = np.c_[sensed_obdistfeatures_Np_K, sensed_evdistfeatures_Np_K,
                                      sensed_evspeedfeatures_Np_K, sensed_podistfeatures_Np_K,
                                      sensed_pospeedfeatures_Np_K, sensed_pudistfeatures_Np_K,
                                      sensed_puspeedfeatures_Np_K]

        # Move objects
        self.evadersx_Ne_2 += self.evadersv_Ne_2
        self.poisonx_Npo_2 += self.poisonv_Npo_2

        # Bounce object if it hits a wall
        self.evadersv_Ne_2[np.clip(self.evadersx_Ne_2, 0, 1) != self.evadersx_Ne_2] *= -1
        self.poisonv_Npo_2[np.clip(self.poisonx_Npo_2, 0, 1) != self.poisonx_Npo_2] *= -1

        obslist = []
        for inp in range(self.n_pursuers):
            obslist.append(
                np.concatenate([sensorfeatures_Np_K_O[inp, ...].ravel(), [float((
                    is_colliding_ev_Np_Ne[inp, :]).sum() > 0), float((is_colliding_po_Np_Npo[inp, :]
                                                                     ).sum() > 0)], [inp + 1]]))

        done = self.is_terminal
        info = None
        return obslist, rewards, done, info

    def render(self, screen_size=800, rate=10):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Obstacles
        for iobs, obstaclex_2 in enumerate(self.obstaclesx_No_2):
            assert obstaclex_2.shape == (2,)
            color = (128, 128, 0)
            cv2.circle(img, tuple((obstaclex_2 * screen_size).astype(int)),
                       int(self.obstacle_radius * screen_size), color, -1, lineType=cv2.CV_AA)
        # Pursuers
        for ipur, pursuerx_2 in enumerate(self.pursuersx_Np_2):
            assert pursuerx_2.shape == (2,)
            for k in range(self.n_sensors):
                color = (0, 0, 0)
                cv2.line(img, tuple((pursuerx_2 * screen_size).astype(int)),
                         tuple(((pursuerx_2 + self.sensor_range[ipur] * self.sensor_vecs_Np_K_2[
                             ipur, k, :]) * screen_size).astype(int)), color, 1, lineType=cv2.CV_AA)
                cv2.circle(img, tuple((pursuerx_2 * screen_size).astype(int)),
                           int(self.radius * screen_size), (255, 0, 0), -1, lineType=cv2.CV_AA)
        # Evaders
        for iev, evaderx_2 in enumerate(self.evadersx_Ne_2):
            color = (0, 255, 0)
            cv2.circle(img, tuple((evaderx_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), color, -1, lineType=cv2.CV_AA)
        # Poison
        for ipo, poisonx_2 in enumerate(self.poisonx_Npo_2):
            color = (0, 0, 255)
            cv2.circle(img, tuple((poisonx_2 * screen_size).astype(int)),
                       int(self.radius * screen_size), color, -1, lineType=cv2.CV_AA)

        opacity = 0.4
        bg = np.ones((screen_size, screen_size, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Waterworld', img)
        cv2.waitKey(rate)


if __name__ == '__main__':
    env = MAWaterWorld(3, 5)
    obs = env.reset()
    while True:
        obs, rew, _, _ = env.step(env.np_random.randn(6) * .5)
        if rew.sum() > 0:
            print(rew)
        env.render()
