

from utils import *
import time

# Environments to be tested on
env_name = 'InvertedPendulum-v1'
# env_name = 'Pendulum-v0'
# env_name = 'HalfCheetah-v1'

GAMMA = 0.99
BATCH_SIZE = 128

MAX_STEPS = 1000
NUM_EPISODES = 12000


env = gym.make(env_name)
env = NormalizeAction(env)

state = env.reset()

solution = load()



for t in range(MAX_STEPS):
    action = solution.get_action(state)
    next_state, reward, done, info = env.step(action)
    time.sleep(0.05)
    print(t)
    if done:
        break
    state = next_state
    env.render()
env.close()