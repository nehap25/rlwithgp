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
#import pybullet as p 
#import pybullet_data
#from simple_world.constants import DIM, ORI, FRICTION, GRIPPER_ORI, GRIPPER_X, GRIPPER_Y, GRIPPER_Z, MASS, MAX_FRICTION, MAX_MASS, MIN_FRICTION, MIN_MASS, STEPS, WLH
#from simple_world.utils import Conf, Pose, Robot, close_enough, create_object, create_stack, eul_to_quat, get_full_aabb, get_pose, get_yaw, rejection_sample_aabb, rejection_sample_region, sample_aabb, set_conf, set_pose, step_simulation
#from simple_world.primitives import get_push_conf, move, push
#from gen_data import *
#full_pose, robot, objects,use_gui,goal_pose,mass_bool=setup(True)
#for action in actions:
#  robot,objects,use_gui,reward=step(action[0],action[1],robot,objects,goal_pose,use_gui)
########################################################


"""
Potential Questions for authors: 
- how to define epsilon, delta, Ns, discount, Rmax in Experiment 1
- GP Update Equations 
- Reward function parameters 
"""

#Parameters to define 
lipschitz = 9
s_t = (0, 0)
current = s_t
dist_per_step = 0.1
actions = [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
noise_var = 0.1
timesteps = 200
states = [(i/10, j/10) for i in range(0, 10) for j in range(0, 10)]
rbf_theta = 0.05

#Accuracy parameters 
epsilon = 
delta = 

#Reward function parameters 
discount = 0.99
####CHANGE BELOW IF YOU CHANGE REWARD FUNCTION####
Rmax = 2
Vmax = Rmax*timesteps
##################################################

def covering_number(states, r, dist_per_step):
	area=math.pi*r*r
	return math.ceil(1/area)

Ns =  covering_number(states, epsilon*(1 - discount)/(3*lipschitz), dist_per_step)# N_S (ε(1−γ)/(3lipschitz))



#Sensitivity analysis of k, (sigma_tol)^2, e_1
multiplicative_factor = epsilon*(1-discount)/(3*lipschitz)
k = len(A)*Ns*multiplicative_factor*((3*Rmax/((1 - discount)**2)*epsilon) + 1)
var_threshold_num = 2*noise_var*(epsilon**2)*((1 - discount)**4)
var_threshold_denom = 9*(Rmax**2)
log_val=log(multiplicative_factor*len(A)*Ns*(1 + k)*6/delta)
var_threshold_denom*=log_val
var_threshold = var_threshold_num/var_threshold_denom
epsilon_one = epsilon*(1 - discount)/3




class GP:

	def __init__(self, means, kernel, states):
		self.mean = {}
		for i in range(len(states)):
			self.mean[states[i]] = mean[i]
		self.kernel = kernel
		self.covar = kernel(states, states, gamma=1/(2*(rbf_theta**2)))
		self.states = states

	def update(self, state, reward_val):
		K_ss = self.kernel(state, state, gamma=1/(2*(rbf_theta**2))) 
		K_s = self.kernel(self.states,state, gamma=1/(2*(rbf_theta**2))) 
		K = self.kernel(self.states, self.states, gamma=(1/(2*(rbf_theta**2)))) + noise_var*np.eye(len(self.states))
		K_inv = inv(K)
		mu_s = K_s.T.dot(K_inv).dot([reward_val])
		for i in range(len(states)):
			self.mean[states[i]] = mu_s[i]
		self.covar = K_ss - K_s.T.dot(K_inv).dot(K_s)

	def mean(self, state):
		return self.mean[state]

	def variance(self, state):
		state_ind = states.index(state)
		return self.covar[state_ind][state_ind]


#Supporting functions 



#Calculates distances between two states (based on L1 metric for experiment 1)
def d(s, si):
	distance = 0
	for i in range(len(s)):
		distance += abs(s[i] - si[i])
	return distance

#Equation 7
def Q(s, a, Q_dict):
	currentMin = float('inf')
	for (si, ai) in Q_dict:
		if ai == a:
			mu = Q_dict[(si, ai)]
			total = mu + lipschitz*d(s, si)
			currentMin = min(currentMin, total)
	return min(currentMin, Vmax)

#Line 6 of algorithm 
def argmax_action(Q_dict, s_t):
	a= None
	currentMax = float('-inf')
	for action in actions:
		if Q(s_t, action, Q_dict) > currentMax:
			currentMax = Q(s_t, action, Q_dict)
			a = action
	return a
def get_reward_v1(s_t):
	if ((1-s_t[0])**2 + (1-s_t[1])**2)**0.5 > 0.15:
		return 0
	return 1, 
def get_reward_v2(s_t):
	return 2 - ((1-s_t[0])**2 + (1-s_t[1])**2)**0.5 


Q_dict = {}
GP_actions = {}
for action in actions:
	means = [Rmax/(1 - discount) for i in range(len(states))]
	GP_actions[action] = GP(means, rbf_kernel, states)

for t in timesteps:
	a_t = argmax_action(Q_dict, s_t)
	s_t = np.add(s_t, a_t)
	r_t = get_reward_v2(s_t) 
	q_t = r_t + discount*max(Q(s_t, a, Q_dict) for a in actions)
	sigma_one_squared = GP_actions[a_t].variance(s_t)
	if sigma_one_squared > var_threshold:
		GP_actions[a_t].update(s_t, q_t) 
	sigma_two_squared = GP_actions[a_t].variance(s_t)
	mean = GP_actions[a_t].mean(s_t)
	if sigma_one_squared > var_threshold and var_threshold >= sigma_two_squared and Q(s_t, a_t, Q_dict) - mean > 2*epsilon_one:
		new_mean = mean + epsilon_one
		selected_keys = []
		for (sj, aj) in Q_dict:
			if new_mean + lipschitz*d(sj, s_t) <= Q_dict[(sj, aj)]:
				selected_keys.append((sj, aj))
		for key in selected_keys:
			del Q_dict[key]
		Q_dict[(s_t, a_t)] = new_mean
		for action in actions:
			new_mu = [Q(s, action, Q_dict) for s in states]
			GP_actions[action] = GP(new_mu, rbf_kernel, states)

