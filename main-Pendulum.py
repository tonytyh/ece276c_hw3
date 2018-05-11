
from ddpg import *
import time

# Environments to be tested on
# env_name = 'InvertedPendulum-v1'
env_name = 'Pendulum-v0'
# env_name = 'HalfCheh-v1'

GAMMA = 0.99
BATCH_SIZE = 128

MAX_STEPS = 1000
NUM_EPISODES = 12000

logging_interval = 100
animate_interval = logging_interval * 5

VISUALIZE = False




env = gym.make(env_name)
env = NormalizeAction(env)  # remap action values for the environment
avg_val = 0

# for plotting
running_rewards_ddpg = []
step_list_ddpg = []
step_counter = 0

# set term_condition for early stopping according to environment being used
term_condition = -150  # Pendulum
ddpg = DDPG(act_dim = env.action_space.shape[0], obs_dim = env.observation_space.shape[0],critic_lr=1e-3, actor_lr=1e-4, gamma = GAMMA, batch_size = BATCH_SIZE)

for itr in range(NUM_EPISODES):
    state = env.reset()  # get initial state
    animate_this_episode = (itr % animate_interval == 0) and VISUALIZE
    total_reward = 0

    while True:
        ddpg.noise.reset()

        if animate_this_episode:
            env.render()
            time.sleep(0.05)

        action = ddpg.get_action_with_noise(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        ddpg.replayBuffer.remember(state, action, reward, next_state, done)
        state = next_state

        # use actor to get action, add ddpg.noise.step() to action
        # remember to put NN in eval mode while testing (to deal with BatchNorm layers) and put it back
        # to train mode after you're done getting the action

        # step action, get next state, reward, done (keep track of total_reward)
        # populate ddpg.replayBuffer

        ddpg.train()
        step_counter += 1

        if done:
            break
    if avg_val > term_condition:
        break

    running_rewards_ddpg.append(total_reward)  # return of this episode
    step_list_ddpg.append(step_counter)

    avg_val = avg_val * 0.95 + 0.05 * running_rewards_ddpg[-1]
    print("Average value: {} for episode: {}".format(avg_val, itr))