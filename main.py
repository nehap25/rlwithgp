import numpy as np

import syntheticChrissAlmgren as sca
from syntheticChrissAlmgren import SINGLE_STEP_VARIANCE

from ddpg_agent import Agent
import ddpg_agent
import math
from math import log 
from numpy.linalg import inv
from collections import deque

# Create simulation environment
env = sca.MarketEnvironment()

# Set the liquidation time
lqt = 60

# Set the number of trades
n_trades = 60

# Set trader's risk aversion
tr = 1e-6

# Set the number of episodes to run the simulation
episodes = 10000

shortfall_hist = np.array([])
shortfall_deque = deque(maxlen=100)

# Parameters to define
lipschitz = 9

# Accuracy parameters--UNKNOWN
epsilon = 1
noise_var = SINGLE_STEP_VARIANCE
delta = 0.01

# Reward function parameters
Rmax = 704723.2174653504
Vmax = Rmax*60
discount = ddpg_agent.GAMMA

actions = [i/100.0 for i in range(101)]
num_actions = len(actions)

# Initialize Feed-forward DNNs for Actor and Critic models. 
agent = Agent(env.observation_space_dimension(), env.action_space_dimension(), actions, random_seed=0)

def covering_number(r):
    area = math.pi*r*r
    return math.ceil(1/area)

Ns = covering_number(epsilon*(1 - discount)/(3*lipschitz))  # N_S (ε(1−γ)/(3lipschitz))

k = num_actions*Ns*(3*Rmax/(((1 - discount)**2)*epsilon) + 1)
var_threshold_num = 2*noise_var*(epsilon**2)*((1 - discount)**4)
var_threshold_denom = 9*(Rmax**2)
log_val = log(num_actions*Ns*(1 + k)*6/delta)
var_threshold_denom *= log_val
var_threshold = var_threshold_num/var_threshold_denom
epsilon_one = epsilon*(1 - discount)/3

for episode in range(episodes): 
    cur_state = env.reset(seed = episode, liquid_time = lqt, num_trades = n_trades, lamb = tr)
    env.start_transactions()

    for i in range(n_trades + 1):
        final_a, actual_a = agent.act(cur_state)
        new_state, reward, done, info = env.step(final_a)
        cur_state = tuple(new_state.tolist())

        if info.done:
            print("GOT HERE")
            shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
            shortfall_deque.append(info.implementation_shortfall)
            break

        q_t = reward + discount*max(agent.Q(cur_state, a) for a in actions)
        sigma_one_squared = agent.GP_actions[actual_a].variance(cur_state)
        if sigma_one_squared > var_threshold:
            if (cur_state,actual_a) in agent.Q_dict:
                agent.GP_actions[actual_a].update(cur_state, q_t - agent.Q_dict[(cur_state,actual_a)])
            else:
                agent.GP_actions[actual_a].update(cur_state, q_t)
        sigma_two_squared = agent.GP_actions[actual_a].variance(cur_state)
        a_t_mean = agent.GP_actions[actual_a].mean_state(cur_state)
        print(i, sigma_two_squared, var_threshold)
        if sigma_one_squared > var_threshold and var_threshold >= sigma_two_squared and agent.Q(cur_state, actual_a) - a_t_mean > 2*epsilon_one:
            print("IN HERE")
            new_mean = a_t_mean + epsilon_one 
            selected_keys = []
            for (sj, aj) in agent.Q_dict:
                if new_mean + lipschitz*agent.d(sj, cur_state) <= agent.Q_dict[(sj, aj)]:
                    selected_keys.append((sj, aj))
            for key in selected_keys:
                del agent.Q_dict[key]
            if (cur_state,actual_a) in agent.Q_dict:
                agent.Q_dict[(cur_state, actual_a)] = agent.Q_dict[(cur_state, actual_a)] + new_mean
            else:
                agent.Q_dict[(cur_state, actual_a)] =  new_mean
            for action in actions:
                agent.GP_actions[action] = GP("Q_MEAN", rbf_kernel, action, Q_dict)
        
    if (episode + 1) % 100 == 0: # print average shortfall over last 100 episodes
        print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))        

# print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(np.mean(shortfall_hist)))