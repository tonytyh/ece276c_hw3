import gym
import torch
import numpy as np
from reinforce import *
import os
from torch.autograd import Variable




# Select Environment

#discrete environment:
env_name='CartPole-v0'

#continous environments:
#env_name='InvertedPendulum-v2'
#env_name = 'HalfCheetah-v2'

# env_name='InvertedPendulum-v1'

# Make the gym environment
env = gym.make(env_name)
visualize = False
animate=visualize

learning_rate = 1e-3
logging_interval = 100

max_path_length=None


# Set random seeds
seed=0
torch.manual_seed(seed)
np.random.seed(seed)

# Saving parameters
logdir='./REINFORCE/'

#
# if visualize:
#     if not os.path.exists(logdir):
#         os.mkdir(logdir)
#     env = wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%animate_interval==0)
# env._max_episodes_steps = min_timesteps_per_batch


# Is this env continuous, or discrete?
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]

# Maximum length for episodes
max_path_length = max_path_length or env.spec.max_episode_steps

# Make network object (remember to pass in appropriate flags for the type of action space in use)
#net = mlp(*args)
net = mlp(obs_dim, 50, act_dim).type(torch.FloatTensor)

# Make optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

# take optimizer step

n_iter = 1000
min_timesteps_per_batch = 2000  # sets the batch size for updating network
avg_reward = 0
avg_rewards = []
step_list_reinforce = []
total_steps = 0
episodes = 0





for itr in range(n_iter):  # loop for number of optimization steps
    paths = []
    steps = 0

    while True:  # loop to get enough timesteps in this batch --> if episode ends this loop will restart till steps reaches limit
        ob = env.reset()
        obs, acs, rews, log_odds = [], [], [], []

        while True:  # loop for episode inside batch
            # if animate_this_episode:
            #     env.render()
            #     time.sleep(0.05)

            state = Variable(torch.from_numpy(ob).float().unsqueeze(0))
            obs.append(ob)
            # get parametrized policy distribution from net using current state ob

            # sample action and get log-probability (log_odds) from distribution
            net.eval()
            logit = net(state)
            net.train()

            action, log_prob = sample_action(logit,discrete)
            acs.append(action)
            log_odds.append(log_prob)
            # step environment, record reward, next state

            ob , reward, done, info = env.step(action)
            # print(ob)

            # append to obs, acs, rewards, log_odds
            rews.append(reward)

            # if done, restart episode till min_timesteps_per_batch is reached

            steps += 1

            if done:
                episodes = episodes + 1
                break


        path = {"observation": obs,
                "reward": np.array(rews),
                "action": (acs),
                "log_odds": log_odds}

        paths.append(path)



        if steps > min_timesteps_per_batch:
            break

    update_policy(paths, optimizer)  # use all complete episodes (a batch of timesteps) recorded in this itr to update net

    if itr == 0:
        avg_reward = path['reward'].sum()
    else:
        avg_reward = avg_reward * 0.95 + 0.05 * path['reward'].sum()
    print("itr[%d]    %s" % (itr, avg_reward))
    if avg_reward > 300:
        break

    total_steps += steps
    avg_rewards.append(avg_reward)
    step_list_reinforce.append(total_steps)
    # if itr % logging_interval == 0:
    # print('Average reward: {}'.format(avg_reward))


env.close()

plt.plot(avg_rewards)
plt.title('Training reward for <env> over multiple runs ')
plt.xlabel('Iteration')
plt.ylabel('Average reward')

