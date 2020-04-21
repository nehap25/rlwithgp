import numpy as np
import random
import copy
from numpy.linalg import inv
from collections import namedtuple, deque
from sklearn.metrics.pairwise import rbf_kernel

from syntheticChrissAlmgren import SINGLE_STEP_VARIANCE

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy.linalg as LA

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.01           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rbf_theta = 0.05
noise_var = SINGLE_STEP_VARIANCE
####CHANGE BELOW IF YOU CHANGE REWARD FUNCTION####
Rmax = 704723.2174653504
Vmax = Rmax*60
discount = GAMMA

class GP:

    def __init__(self, mean, kernel, action, Q_dict={}):
        self.mean = mean
        self.kernel = kernel
        self.action = action
        self.Q_dict = Q_dict
        self.x_values = []
        self.y_values = []

    def update(self, state, reward_val): 
        self.x_values.append(state)
        self.y_values.append(reward_val)

    def mean_state(self, state): 
        new_mean = self.mean
        if self.mean == "Q_MEAN" and len(self.x_values) == 0:
            new_mean = Q(state, self.action, self.Q_dict)
        if self.x_values != []:
            new_state = np.array(list(state)).reshape(1, -1)
            KXstate = self.kernel(np.array(self.x_values), new_state,
                                  gamma=1/(2*(rbf_theta**2)))
            KXX = inv(self.kernel(np.array(self.x_values), np.array(self.x_values), gamma=1/(2*(rbf_theta**2))) +
                      noise_var * np.eye(len(self.x_values)))
            new_mean = np.matmul(np.matmul(KXstate.T, KXX), np.array(self.y_values))[0]
        return new_mean

    def variance(self, state):
        new_state = np.array(list(state)).reshape(1, -1)
        if self.x_values != []:
            Kxx = self.kernel(new_state, new_state, gamma=1/(2*(rbf_theta**2)))
            KXstate = self.kernel(np.array(self.x_values), new_state,
                                  gamma=1/(2*(rbf_theta**2)))
            KXX = inv(self.kernel(np.array(self.x_values), np.array(self.x_values), gamma=1 /
                                  (2*(rbf_theta**2))) + noise_var*np.eye(len(self.x_values)))
            new_variance = Kxx[0][0] - np.matmul(np.matmul(KXstate.T, KXX), KXstate)[0][0]
            state_variance = new_variance
        else:
            state_variance = self.kernel(new_state, new_state, gamma=1/(2*(rbf_theta**2)))[0][0]
        return state_variance

class Agent():
    """Interacts with and learns from the environment."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def __init__(self, state_size, action_size, actions, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.Q_dict = {}
        self.GP_actions = {}
        self.actions = actions
        for action in actions:
            self.GP_actions[action] = GP(Rmax/(1 - discount), rbf_kernel, action)

    def act(self, state):
        noise_sample = self.noise.sample().item()
        final_a = None
        actual_a = None
        currentMax = float('-inf')
        for action in self.actions:
            new_a = action + noise_sample
            new_a = np.clip(new_a, 0, 1)
            new_a = round(new_a, 3)
            if self.Q(state, action) > currentMax:
                currentMax = self.Q(state, action)
                final_a = new_a
                actual_a = action
        return final_a, actual_a


    def reset(self):
        self.noise.reset()    

    # Equation 7
    def Q(self, s, a):
        currentMin = float('inf')
        for (si, ai) in self.Q_dict:
            if ai == a:
                mu = self.Q_dict[(si, ai)]
                total = mu + lipschitz*d(s, si)
                currentMin = min(currentMin, total)
        return min(currentMin, Vmax)

    # Calculates distances between two states (based on L2 Metric)
    def d(s, si):
        return LA.norm(s - si)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
