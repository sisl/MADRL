import numpy as np
from gym import spaces
import scipy.spatial.distance as ssd

import matplotlib.pyplot as plt


class CentralizedWaterWorld(object):
    def __init__(self, n_pursuers, n_evaders, n_coop=2, radius=0.015, ev_speed=0.01,
                 n_poison=10, poison_speed=0.01,
                 n_sensors=30, sensor_range=0.2, action_scale=0.01,
                 poison_reward=-1., food_reward=1., control_penalty=-1., **kwargs):
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_coop = n_coop
        self.n_poison = n_poison
        self.poison_speed = poison_speed
        self.radius = radius
        self.ev_speed = ev_speed
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range
        self.action_scale = action_scale
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.control_penalty = control_penalty

        # Number of observation coordinates from each senser
        self.sensor_obscoord = 4
        self.obscoord_from_sensors = n_sensors * self.sensor_obscoord
        self._obs_dim = self.obscoord_from_sensors + 2 #2 for type


    @property
    def observation_space(self):
        return spaces.Box(low=0, high=self.sensor_range, shape=(self.n_pursuers*self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=-10, high=10, shape=(self.n_pursuers*2))

    def reset(self):
        # Initialize pursuers
        self.pursuersx_Np_2 = np.random.rand(self.n_pursuers, 2)
        self.pursuersv_Np_2 = np.zeros((self.n_pursuers, 2))

        # Sensors
        angles_K = np.linspace(0., 2.*np.pi, self.n_sensors+1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]
        self.sensor_vecs_Np_K_2 = np.tile(sensor_vecs_K_2, (self.n_pursuers, 1, 1))

        # Initialize evaders
        self.evadersx_Ne_2 = np.random.rand(self.n_evaders, 2)
        self.evadersv_Ne_2 = (np.random.rand(self.n_evaders, 2)-.5)*self.ev_speed # Random speeds TODO policy?

        # Initialize poisons
        self.poisonx_Npo_2 = np.random.rand(self.n_poison, 2)
        self.poisonv_Npo_2 = (np.random.rand(self.n_poison, 2)-.5)*self.poison_speed # Random speeds

        return self.step(np.zeros((self.n_pursuers, 2)))

    @property
    def is_terminal(self):
        return False

    def caught(self, is_colliding_Np_Ne, n_coop):
        n_collisions_Ne = is_colliding_Np_Ne.sum(axis=0)
        is_caught_Ne = n_collisions_Ne >= n_coop
        catches = is_caught_Ne.sum()
        return catches, is_caught_Ne

    def step(self, action_Np2):
        action_Np_2 = action_Np2.reshape(self.n_pursuers, 2)
        # Players
        actions_Np_2 = action_Np_2 * self.action_scale

        reward = 0.
        assert action_Np_2.shape == (self.n_pursuers, 2)

        self.pursuersv_Np_2 += actions_Np_2
        self.pursuersx_Np_2 += self.pursuersv_Np_2

        # Penalize large actions
        reward += self.control_penalty * (actions_Np_2**2).sum()

        # Players stop on hitting a wall
        clippedx_Np_2 = np.clip(self.pursuersx_Np_2, 0, 1)
        self.pursuersv_Np_2[self.pursuersx_Np_2 != clippedx_Np_2] = 0
        self.pursuersx_Np_2 = clippedx_Np_2

        # Find collisions
        # Evaders
        evdists_Np_Ne = ssd.cdist(self.pursuersx_Np_2, self.evadersx_Ne_2)
        is_colliding_ev_Np_Ne = evdists_Np_Ne <= self.radius*2
        # num_collisions depends on how many needed to catch an evader
        ev_catches, ev_caught_Ne = self.caught(is_colliding_ev_Np_Ne, self.n_coop)
        # Poisons
        podists_Np_Npo = ssd.cdist(self.pursuersx_Np_2, self.poisonx_Npo_2)
        is_colliding_po_Np_Npo = podists_Np_Npo <= self.radius*2
        num_poison_collisions = is_colliding_po_Np_Npo.sum()

        # Find sensed objects
        
        # Evaders
        relpos_ev_Ne_Np_2 = self.evadersx_Ne_2[:,None,:] - self.pursuersx_Np_2
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(self.sensor_vecs_Np_K_2[inp,...].dot(relpos_ev_Ne_Np_2[:,inp,:].T))
        sensorvals_Np_K_Ne = np.c_[sensorvals]
        sensorvals_Np_K_Ne[(sensorvals_Np_K_Ne < 0) | (sensorvals_Np_K_Ne > self.sensor_range) | ((relpos_ev_Ne_Np_2**2).sum(axis=2).T[:,None,...] - sensorvals_Np_K_Ne**2 > self.radius**2)] = np.inf # TODO: check

        # Poison
        relpos_po_Npo_Np_2 = self.poisonx_Npo_2[:,None,:] - self.pursuersx_Np_2
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(self.sensor_vecs_Np_K_2[inp,...].dot(relpos_po_Npo_Np_2[:,inp,:].T))
        sensorvals_Np_K_Npo = np.c_[sensorvals]
        sensorvals_Np_K_Npo[(sensorvals_Np_K_Npo < 0) | (sensorvals_Np_K_Npo > self.sensor_range) | ((relpos_po_Npo_Np_2**2).sum(axis=2).T[:,None,...] - sensorvals_Np_K_Npo**2 > self.radius**2)] = np.inf # TODO: check

        # TODO (other pursuers)
        # dist features
        closest_ev_idx_Np_K = np.argmin(sensorvals_Np_K_Ne, axis=2)
        sensedmask_ev_Np_K = np.isfinite(closest_ev_idx_Np_K)
        sensed_evdistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_evdistfeatures_Np_K[sensedmask_ev_Np_K] = closest_ev_idx_Np_K[sensedmask_ev_Np_K]
        closest_po_idx_Np_K = np.argmin(sensorvals_Np_K_Npo, axis=2)
        sensedmask_po_Np_K = np.isfinite(closest_po_idx_Np_K)
        sensed_podistfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensed_podistfeatures_Np_K[sensedmask_po_Np_K] = closest_po_idx_Np_K[sensedmask_po_Np_K]

        # speed features
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(self.sensor_vecs_Np_K_2[inp,...].dot((self.evadersv_Ne_2 - self.pursuersv_Np_2[inp,...]).T))
        sensed_evspeed_Np_K_Ne = np.c_[sensorvals]
        sensed_evspeedfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(sensed_evspeed_Np_K_Ne[inp,:,:][np.arange(self.n_sensors), closest_ev_idx_Np_K[inp,:]])
        sensed_evspeedfeatures_Np_K[sensedmask_ev_Np_K] = np.c_[sensorvals][sensedmask_ev_Np_K]

        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(self.sensor_vecs_Np_K_2[inp,...].dot((self.poisonv_Npo_2 - self.pursuersv_Np_2[inp,...]).T))
        sensed_pospeed_Np_K_Npo = np.c_[sensorvals]
        sensed_pospeedfeatures_Np_K = np.zeros((self.n_pursuers, self.n_sensors))
        sensorvals = []
        for inp in range(self.n_pursuers):
            sensorvals.append(sensed_pospeed_Np_K_Npo[inp,:,:][np.arange(self.n_sensors), closest_po_idx_Np_K[inp,:]])
        sensed_pospeedfeatures_Np_K[sensedmask_ev_Np_K] = np.c_[sensorvals][sensedmask_po_Np_K]

        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        self.evadersx_Ne_2[ev_caught_Ne,:] = np.random.rand(ev_catches, 2)
        self.evadersv_Ne_2[ev_caught_Ne,:] = (np.random.rand(ev_catches, 2)-.5)*self.ev_speed

        po_catches, po_caught_Npo = self.caught(is_colliding_po_Np_Npo, 1)
        self.poisonx_Npo_2[po_caught_Npo,:] = np.random.rand(po_catches, 2)
        self.poisonv_Npo_2[po_caught_Npo,:] = (np.random.rand(po_catches, 2)-.5)*self.poison_speed

        # Update reward based on these collisions
        reward += ev_catches*self.food_reward + po_catches*self.poison_reward

        # Add features together
        sensorfeatures_Np_K_O = np.c_[sensed_evdistfeatures_Np_K, sensed_evspeedfeatures_Np_K, sensed_podistfeatures_Np_K, sensed_pospeedfeatures_Np_K]

        # Move objects
        self.evadersx_Ne_2 += self.evadersv_Ne_2
        self.poisonx_Npo_2 += self.poisonv_Npo_2

        # Bounce object if it hits a wall
        self.evadersv_Ne_2[np.clip(self.evadersx_Ne_2, 0, 1) != self.evadersx_Ne_2] *= -1
        self.poisonv_Npo_2[np.clip(self.poisonx_Npo_2, 0, 1) != self.poisonx_Npo_2] *= -1

        obslist = []
        for inp in range(self.n_pursuers):
            obslist.append(np.concatenate([sensorfeatures_Np_K_O[inp,...].ravel(), [float((is_colliding_ev_Np_Ne[inp,:]).sum() > 0), float((is_colliding_po_Np_Npo[inp,:]).sum() > 0)]]))
        obs = np.c_[obslist].ravel()
        assert obs.shape == self.observation_space.shape
        done = self.is_terminal
        info = None
        return obs, reward, done, info

    def render(self, screen_size=800):
        import cv2
        img = np.empty((screen_size, screen_size, 3), dtype=np.uint8)
        img[...] = 255
        # Pursuers
        for ipur, pursuerx_2 in enumerate(self.pursuersx_Np_2):
            assert pursuerx_2.shape == (2,)
            for k in range(self.n_sensors):
                color = (0,0,0)
                cv2.line(
                    img,
                    tuple((pursuerx_2*screen_size).astype(int)),
                    tuple(((pursuerx_2+self.sensor_range*self.sensor_vecs_Np_K_2[ipur,k,:])*screen_size).astype(int)),
                    color,
                    1, lineType=cv2.CV_AA
                )
                cv2.circle(
                    img,
                    tuple((pursuerx_2*screen_size).astype(int)),
                    int(self.radius*screen_size),
                    (255,0,0),
                    -1, lineType=cv2.CV_AA)
        for iev, evaderx_2 in enumerate(self.evadersx_Ne_2):
            color = (0,255,0)
            cv2.circle(
                img,
                tuple((evaderx_2*screen_size).astype(int)),
                int(self.radius*screen_size),
                color,
                -1, lineType=cv2.CV_AA
            )

        for ipo, poisonx_2 in enumerate(self.poisonx_Npo_2):
            color = (0,0,255)
            cv2.circle(
                img,
                tuple((poisonx_2*screen_size).astype(int)),
                int(self.radius*screen_size),
                color,
                -1, lineType=cv2.CV_AA
            )
        cv2.imshow('Waterworld', img)
        cv2.waitKey(10)


if __name__ == '__main__':
    env = CentralizedWaterWorld(3, 5)
    obs = env.reset()
    while True:
        env.step(np.random.randn(3, 2)*.5)
        env.render()
