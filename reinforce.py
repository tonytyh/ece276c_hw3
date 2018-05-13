import torch

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import gym
import numpy as np
# ----------------------------------------------------
# Policy parametrizing model, MLP
# ----------------------------------------------------
# 1 or 2 hidden layers with a small number of units per layer (similar to DQN)
# use ReLU for hidden layer activations
# softmax as activation for output if discrete actions, linear for continuous control
# for the continuous case, output_dim=2*act_dim (each act_dim gets a mean and std_dev)

class mlp(nn.Module):
    def __init__(self,obs_dim, hidden_size,act_dim):
        super(mlp,self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x),dim = 1)

        return action_probs


# policy = mlp(env.ob)


def sample_action(logit, discrete):
    # logit is the output of the softmax/linear layer
    # discrete is a flag for the environment type
    # Hint: use Categorical and Normal from torch.distributions to sample action and get the log-probability
    # Note that log_probability in this case translates to ln(\pi(a|s))

    m = Categorical(logit)
    action = m.sample()
    log_odds = m.log_prob(action)
    return action.item(), log_odds




def rewardtogo(rewards):
    R = 0
    arr = []
    for r in rewards[::-1]:
        # for each time-step
        R = r + R
        arr.insert(0, R)

    return arr



def update_policy(paths, optimizer):
    # paths: a list of paths (complete episodes, used to calculate return at each time step)
    # net: MLP object

    num_paths = len(paths)
    rew_cums = []
    log_odds = []

    # for path in paths:
    # rew_cums should record return at each time step for each path
    # log_odds should record log_odds obtained at each timestep of path
    # calculated as "reward to go"

    for path in paths:
        rew_cums += rewardtogo(path["reward"])
        log_odds += path["log_odds"]

    rew_cums = np.array(rew_cums)
    log_odds = np.array(log_odds)
    rew_cums = (rew_cums - rew_cums.mean()) / (rew_cums.std() + 1e-5)  # create baseline
    policy_loss = [(- log_odd * reward) for log_odd, reward in zip(log_odds, rew_cums)]
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).mean()
    loss.backward()
    optimizer.step()
