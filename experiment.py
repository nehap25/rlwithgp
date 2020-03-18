from math import log


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

for action in actions:



for t in timesteps:





