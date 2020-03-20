from math import log
import os
import pybullet as p #DEPENDENCY
import pybullet_data
from numpy import float32
from numpy import array
from simple_world.constants import DIM, ORI, FRICTION, GRIPPER_ORI, GRIPPER_X, GRIPPER_Y, GRIPPER_Z, MASS, MAX_FRICTION, MAX_MASS, MIN_FRICTION, MIN_MASS, STEPS, WLH
from simple_world.utils import Conf, Pose, Robot, close_enough, create_object, create_stack, eul_to_quat, get_full_aabb, get_pose, get_yaw, rejection_sample_aabb, rejection_sample_region, sample_aabb, set_conf, set_pose, step_simulation
from simple_world.primitives import get_push_conf, move, push
from gen_data import *
import time
import math
full_pose, robot, objects,use_gui,goal_pose,mass_bool=setup(True)
#change boolean to False to get rid of visualization
"""
Potential Questions for authors: 
- how to define epsilon, delta, Ns, discount, Rmax in Experiment 1
"""


#Supporting functions 
def covering_number(states, r, dist_per_step):
	return len(states)


#Parameters to define 
lipschitz = 9
start = (0, 0)
dist_per_step = 0.1
actions = [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
noise_var = 0.1
timesteps = 200
states = [(i, j) for i in range(0, 1, 0.1) for j in range(0, 1, 0.1)]

#Accuracy parameters 
epsilon = 
delta = 
Ns =  covering_number(states, epsilon*(1 - discount)/(3*lipschitz), dist_per_step)# N_S (ε(1−γ)/(3lipschitz))

#Reward function parameters 
discount = 
Rmax =

#Sensitivity analysis of k, (sigma_tol)^2, e_1
k = len(A)*Ns*((3*Rmax/((1 - discount)**2)*epsilon) + 1)
var_threshold_num = 2*noise_var*(epsilon**2)*((1 - discount)**4)
var_threshold_denom = 9*(Rmax**2)*log(len(A)*Ns*(1 + k)*6/delta)
var_threshold = var_threshold_num/var_threshold_denom
epsilon_one = epsilon*(1 - discount)/3


##CODE FOR ONE EPISODE##
full_pose, robot, objects,use_gui,goal_pose,mass_bool=setup(True)
for action in actions:
  robot,objects,use_gui,reward=step(action[0],action[1],robot,objects,goal_pose,use_gui)
  print(reward)
########################



