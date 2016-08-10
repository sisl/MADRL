import gym

env = gym.make('BipedalWalker-v2')

o = env.reset()

for i in xrange(50):
    env.render()
    a = env.action_space.sample()
    o, r, done, _ = env.step(a)
    print o, r, done
