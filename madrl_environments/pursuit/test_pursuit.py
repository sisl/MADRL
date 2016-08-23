from pursuit_evade import PursuitEvade
import gym

from utils import *

xs = 10
ys = 10
n_evaders = 6
n_pursuers = 10

map_mat = TwoDMaps.rectangle_map(xs, ys) 

# obs_range should be odd 3, 5, 7, etc
env = PursuitEvade([map_mat], n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=9, n_catch=2, surround=True,
        reward_mech='local')

o = env.reset()

a = [4]*n_pursuers

env.pursuer_layer.set_position(0, 7, 1)
env.pursuer_layer.set_position(1, 8, 0)
env.pursuer_layer.set_position(2, 9, 1)
env.pursuer_layer.set_position(3, 8, 2)


env.pursuer_layer.set_position(4, 0, 2)
env.pursuer_layer.set_position(5, 0, 4)
env.pursuer_layer.set_position(6, 1, 3)

env.pursuer_layer.set_position(7, 3, 4)
env.pursuer_layer.set_position(8, 2, 5)
env.pursuer_layer.set_position(9, 3, 6)

env.evader_layer.set_position(0, 8, 1)
env.evader_layer.set_position(1, 8, 1)

env.evader_layer.set_position(2, 0, 3)
env.evader_layer.set_position(3, 0, 3)

env.evader_layer.set_position(4, 3, 5)
env.evader_layer.set_position(5, 3, 5)


o, r, done, _ = env.step(a)

#map_mat = multi_scale_map(32, 32)
map_mat = multi_scale_map(32, 32, scales=[(4, [0.2,0.3]), (10, [0.1,0.2])])

map_pool = [map_mat]
map_pool = np.load('../../runners/maps/map_pool32.npy')

env = PursuitEvade(map_pool, n_evaders=50, n_pursuers=30, obs_range=11, n_catch=2, surround=True, reward_mech='local',
        sample_maps=True, constraint_window=1.0)
env.reset()
env.render()
