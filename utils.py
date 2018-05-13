import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle

class NormalizeAction(gym.ActionWrapper):
    def action(self, action):

        # tanh outputs (-1,1) from tanh, need to be [action_space.low, action_space.high]
        action = (action + 1) / 2
        action = action * (self.action_space.high - self.action_space.low)
        action = action + self.action_space.low
        return action

    def reverse_action(self, action):
        # reverse of that above
        action = action - self.action_space.low
        action = action / (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action



def weightSync(target_model, source_model, tau = 0.001):
    for parameter_target, parameter_source in zip(target_model.parameters(), source_model.parameters()):
        parameter_target.data.copy_((1 - tau) * parameter_target.data + tau * parameter_source.data)




# class OrnsteinUhlenbeckProcess():
#     def __init__(self, dimension, num_steps = None, mu = 0, theta = 0.15, sigma = 0.2):
#         self.action_dim = dimension
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.X = np.ones(self.action_dim) * self.mu
#         self.num_steps = num_steps
#
#     def init(self):
#         self.X = np.ones(self.action_dim) * self.mu
#
#     def sample(self):
#         dx = self.theta * (self.mu - self.X)
#         dx = dx + self.sigma * np.random.randn(len(self.X))
#         self.X = self.X + dx
#         return self.X

    # def get_sample(self):
    #     data = []
    #     for i in range(self.num_steps):
    #         data.append(self.sample())
    #     return data.copy()


class OrnsteinUhlenbeckProcess():
    def __init__(self, dimension=None, num_steps=None):
        self.theta = 0.15
        self.mu = 0
        self.dt = 0.01
        self.sigma = 0.3
        self.dimension = dimension
        self.reset()

    def sample(self):
        x_new = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(
            size=self.dimension)
        self.x = x_new
        return x_new

    def reset(self):
        self.x = np.zeros((self.dimension))


def save(model):
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)

def load():
    with open('model.pkl','rb') as f:
        model = pickle.load(f)


    return model


if __name__ == "__main__":

    ou = OrnsteinUhlenbeckProcess(1)
    data = []
    for i in range(8000):
        data.append(ou.sample())
    plt.plot(data)
    plt.show()






