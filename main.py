import torch

from utils import *
from ddpg import *

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

solution = DDPG(act_dim = env.action_space.shape[0], obs_dim = env.observation_space.shape[0],critic_lr=1e-3, actor_lr=1e-4, gamma = GAMMA, batch_size = BATCH_SIZE)


for ep in range(NUM_EPISODES):
    state = env.reset()
    G = 0
    count = 0

    for t in range(MAX_STEPS):
        # env.render()
        action = solution.get_action_with_noise(state)
        next_state, reward, done, info = env.step(action)
        G += reward
        solution.replayBuffer.remember(state, action, reward, next_state, done)
        # print(reward)
        count += 1
        state = next_state

        if solution.replayBuffer.size > BATCH_SIZE:
            loss_actor, loss_critic = solution.train()
            # print("[%d]loss: %s %s reward: %s" % (ep, loss_actor, loss_critic, G))

        if done:
            break

        if ep % 100 == 0:
            print("rewards : %d"%G)
    # if ep%100 == 0:


# env.close()
save(solution)







