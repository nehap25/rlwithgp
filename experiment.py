from math import log
import os
import numpy as np
from numpy import float32
from numpy import array
from numpy.linalg import inv

import time
import math
from sklearn.metrics.pairwise import rbf_kernel

####CODE FOR ROBOTICS ENVIRONMENT--UNNECESSARY NOW#####
#######################################################
# import pybullet as p
# import pybullet_data
# from simple_world.constants import DIM, ORI, FRICTION, GRIPPER_ORI, GRIPPER_X, GRIPPER_Y, GRIPPER_Z, MASS, MAX_FRICTION, MAX_MASS, MIN_FRICTION, MIN_MASS, STEPS, WLH
# from simple_world.utils import Conf, Pose, Robot, close_enough, create_object, create_stack, eul_to_quat, get_full_aabb, get_pose, get_yaw, rejection_sample_aabb, rejection_sample_region, sample_aabb, set_conf, set_pose, step_simulation
# from simple_world.primitives import get_push_conf, move, push
# from gen_data import *
# full_pose, robot, objects,use_gui,goal_pose,mass_bool=setup(True)
# for action in actions:
#  robot,objects,use_gui,reward=step(action[0],action[1],robot,objects,goal_pose,use_gui)
########################################################


# Parameters to define
lipschitz = 9
s_t = (0, 0)
actions = [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
noise_var = 0.1
timesteps = 10000
states = [(i/10, j/10) for i in range(0, 10) for j in range(0, 10)]
rbf_theta = 0.05

# Accuracy parameters--UNKNOWN
epsilon = 99
delta = 0.99

# Reward function parameters
discount = 0.99
####CHANGE BELOW IF YOU CHANGE REWARD FUNCTION####
Rmax = 2
Vmax = Rmax*timesteps
##################################################


def covering_number(r):
    area = math.pi*r*r
    return math.ceil(1/area)


Ns = covering_number(epsilon*(1 - discount)/(3*lipschitz))  # N_S (ε(1−γ)/(3lipschitz))


# Sensitivity analysis of k, (sigma_tol)^2, e_1
k = len(actions)*Ns*(3*Rmax/(((1 - discount)**2)*epsilon) + 1)
var_threshold_num = 2*noise_var*(epsilon**2)*((1 - discount)**4)
var_threshold_denom = 9*(Rmax**2)
log_val = log(len(actions)*Ns*(1 + k)*6/delta)
var_threshold_denom *= log_val
var_threshold = var_threshold_num/var_threshold_denom
epsilon_one = epsilon*(1 - discount)/3

class GP:

    def __init__(self, mean, kernel):
        self.mean = mean
        self.kernel = kernel
        self.x_values = []
        self.y_values = []

    def update(self, state, reward_val): 
        self.x_values.append(state)
        self.y_values.append(reward_val)

    def mean_state(self, state): 
        new_mean = self.mean
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


# Calculates distances between two states (based on L1 metric for experiment 1)
def d(s, si):
    distance = 0
    for i in range(len(s)):
        distance += abs(s[i] - si[i])
    return distance

# Equation 7
def Q(s, a, Q_dict):
    currentMin = float('inf')
    for (si, ai) in Q_dict:
        if ai == a:
            mu = Q_dict[(si, ai)]
            total = mu + lipschitz*d(s, si)
            currentMin = min(currentMin, total)
    return min(currentMin, Vmax)

# Line 6 of algorithm
def argmax_action(Q_dict, s_t, noise):
    final_a = None
    actual_a = None
    currentMax = float('-inf')
    for action in actions:
        new_a = action
        if action[0] == 0:
            new_a = (action[0], action[1] + noise)
        else:
            new_a = (action[0] + noise, action[1])
        new_s = np.add(s_t, new_a).tolist()
        new_s = tuple([round(x, 2) for x in new_s])
        if new_s[0] < 0 or new_s[0] > 1 or new_s[1] < 0 or new_s[1] > 1:
            continue
        if Q(s_t, action, Q_dict) > currentMax:
            currentMax = Q(s_t, action, Q_dict)
            final_a = new_a
            actual_a = action
    return final_a, actual_a


def get_reward_v1(s_t):
    if ((1-s_t[0])**2 + (1-s_t[1])**2)**0.5 > 0.15:
        return 0
    return 1,

def get_reward_v2(s_t):
    return 2 - ((1-s_t[0])**2 + (1-s_t[1])**2)**0.5


Q_dict = {}
GP_actions = {}
for action in actions:
    GP_actions[action] = GP(Rmax/(1 - discount), rbf_kernel)

for t in range(timesteps):
    noise = np.random.uniform(-0.01, 0.01)
    final_a, actual_a = argmax_action(Q_dict, s_t, noise)
    s_t = tuple(np.add(s_t, final_a).tolist())
    s_t = tuple([round(i, 2) for i in s_t])
    r_t = get_reward_v2(s_t)
    q_t = r_t + discount*max(Q(s_t, a, Q_dict) for a in actions)
    sigma_one_squared = GP_actions[actual_a].variance(s_t)
    if sigma_one_squared > var_threshold:
        GP_actions[actual_a].update(s_t, q_t)
    sigma_two_squared = GP_actions[actual_a].variance(s_t)
    a_t_mean = GP_actions[actual_a].mean_state(s_t)
    print(t, s_t, sigma_two_squared)
    if sigma_one_squared > var_threshold and var_threshold >= sigma_two_squared and Q(s_t, actual_a, Q_dict) - a_t_mean > 2*epsilon_one:
        new_mean = a_t_mean + epsilon_one
        selected_keys = []
        for (sj, aj) in Q_dict:
            if new_mean + lipschitz*d(sj, s_t) <= Q_dict[(sj, aj)]:
                selected_keys.append((sj, aj))
        for key in selected_keys:
            del Q_dict[key]
        Q_dict[(s_t, actual_a)] = new_mean
        for action in actions:
            new_mu = [Q(s, action, Q_dict) for s in states]
            GP_actions[action] = GP(new_mu, rbf_kernel)
