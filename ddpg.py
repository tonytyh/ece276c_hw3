import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random



GAMMA = 0.99
BATCH_SIZE = 128



# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
# use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def fanin_init(size, fanin = None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)

    return torch.Tensor(size).uniform_(-v, v)


# ----------------------------------------------------
# actor model, MLP
# ----------------------------------------------------
# 2 hidden layers, 400 units per layer, tanh output to bound outputs between -1 and 1

class actor(nn.Module):

# ----------------------------------------------------
# critic model, MLP
# ----------------------------------------------------
# 2 hidden layers, 300 units per layer, ouputs rewards therefore unbounded
# Action not to be included until 2nd layer of critic (from paper). Make sure to formulate your critic.forward() accordingly
    def __init__(self, input_size, output_size):
        super(actor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 400)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(400,400)
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(400,output_size)
        # self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(400)



    def forward(self, state):

        state = state.type(FloatTensor)
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.tanh(self.fc3(x))

        return x


class critic(nn.Module):

    def __init__(self, state_size, action_size,output_size):
        super(critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.output_size = output_size

        self.fc1 = nn.Linear(state_size,300)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(300 + action_size,300)
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(300,output_size)
        # self.fc3.weight.data.uniform_(-0.003,0.003)
        self.bn1 = nn.BatchNorm1d(300)

    def forward(self, state, action):
        state = state.type(FloatTensor)
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(torch.cat([x,action],1))
        x = F.relu(x)
        x = self.fc3(x)

        return x





class DDPG:
    def __init__(self, obs_dim, act_dim, critic_lr=1e-3, actor_lr=1e-5, gamma=GAMMA, batch_size=BATCH_SIZE):
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE

        # actor
        self.actor = actor(input_size=obs_dim, output_size=act_dim).type(FloatTensor)
        self.actor_target = actor(input_size=obs_dim, output_size=act_dim).type(FloatTensor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size=obs_dim, action_size=act_dim, output_size=1).type(FloatTensor)
        self.critic_target = critic(state_size=obs_dim, action_size=act_dim, output_size=1).type(FloatTensor)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-2)

        # critic loss
        self.critic_loss = nn.MSELoss()

        # noise
        self.noise = OrnsteinUhlenbeckProcess(dimension=act_dim)

        # replay buffer
        self.replayBuffer = Replay()

    def train(self):
        # sample from Replay

        if self.replayBuffer.size < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replayBuffer.sample(BATCH_SIZE)

        states =  Variable(torch.from_numpy(states).type(FloatTensor))
        actions = Variable(torch.from_numpy(actions).type(FloatTensor))
        rewards = Variable(torch.from_numpy(rewards).type(FloatTensor))
        next_states = Variable(torch.from_numpy(next_states).type(FloatTensor))
        dones = Variable(torch.from_numpy(dones).type(FloatTensor))



        # update critic (create target for Q function)
        actor_target_next_action = self.actor_target(next_states).detach()

        next_q = torch.squeeze(self.critic_target(next_states,actor_target_next_action).detach())

        y_expected = rewards + self.gamma * next_q * (1. - dones)

        y_predicted = torch.squeeze(self.critic(states, actions))

        loss_critic = self.critic_loss(y_predicted, y_expected)

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()


        pred_action = self.actor(states)
        loss_actor = -1 * (self.critic(states, pred_action)).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()


        # print("loss: %s %s"%(loss_actor.data.numpy(), loss_critic.data.numpy()))


        # critic optimizer and backprop step (feed in target and predicted values to self.critic_loss)

        # update actor (formulate the loss wrt which actor is updated)

        # actor optimizer and backprop step (loss_actor.backward())

        # sychronize target network with fast moving one
        weightSync(self.critic_target, self.critic)
        weightSync(self.actor_target, self.actor)

        return loss_actor.cpu().data.numpy(), loss_critic.cpu().data.numpy()


    def get_action_with_noise(self,state):
        state = Variable(torch.from_numpy(state).unsqueeze(0)).type(FloatTensor)
        self.actor.eval()
        action = self.actor(state).detach()
        self.actor.train()
        new_action = action.cpu().data[0].numpy() + self.noise.sample()
        return new_action
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0)).type(FloatTensor)
        self.actor.eval()
        action = self.actor(state).detach()
        self.actor.train()

        return action.cpu().data[0].numpy()






class Replay:
    def __init__(self,MAX_SIZE = 60000):

        self.buffer = []
        self.size = 0
        self.max_size = MAX_SIZE



    def remember(self, state, action, reward, next_state, done):

        if done:
            done = 1
        else:
            done = 0

        if self.size > self.max_size:
            del self.buffer[0]
            self.size -= 1
        else:
            self.buffer.append([state, action, reward, next_state, done])
            self.size += 1


    def sample(self, batch_size):
        if self.size < batch_size:
            return
        data = random.sample(self.buffer, batch_size)

        states = np.array([x[0] for x in data])
        actions = np.array([x[1] for x in data])
        rewards = np.array([x[2] for x in data])
        next_states = np.array([x[3] for x in data])
        dones = np.array([x[4] for x in data])

        return states, actions, rewards, next_states, dones


if __name__ == "__main__":





    pass