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
        self.sensor_obscoord = 5
        self.obscoord_from_sensors = n_sensors * self.sensor_obscoord
        self._obs_dim = self.obscoord_from_sensors + 2 # 2 for type of catch


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

    def step(self, action_Np_2):
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
        relpos_ev_Ne_2_Np = relpos_ev_Ne_Np_2.transpose(0,2,1)
        sensorvals_Np_K_Ne_Np = self.sensor_vecs_Np_K_2.dot(relpos_ev_Ne_2_Np)
        sensorvals_Np_K_Ne_Np[(sensorvals_Np_K_Ne_Np < 0) | (sensorvals_Np_K_Ne_Np > self.sensor_range) | ((relpos_ev_Ne_2_Np**2).sum(axis=1)[None,...] - sensorvals_Np_K_Ne_Np**2 > self.radius**2)] = np.inf # TODO: check
        # Poison
        relpos_po_Npo_Np_2 = self.poisonx_Npo_2[:,None,:] - self.pursuersx_Np_2
        relpos_po_Npo_2_Np = relpos_po_Npo_Np_2.transpose(0,2,1)
        sensorvals_Np_K_Npo_Np = self.sensor_vecs_Np_K_2.dot(relpos_po_Npo_2_Np)
        sensorvals_Np_K_Npo_Np[(sensorvals_Np_K_Npo_Np < 0) | (sensorvals_Np_K_Npo_Np > self.sensor_range) | ((relpos_po_Npo_2_Np**2).sum(axis=1)[None,...] - sensorvals_Np_K_Npo_Np**2 > self.radius**2)] = np.inf # TODO: check
        
        # TODO
        # dist features

        # speed features
        
        # Process collisions
        # If object collided with required number of players, reset its position and velocity
        # Effectively the same as removing it and adding it back
        
        
        # Update reward based on these collisions
        
        # Add features together
        
        
        # Move objects
        
        # Bounce object if it hits a wall
        
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
    while True:
        obs = env.reset()
        env.render()
