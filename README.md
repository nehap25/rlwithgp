# rlwithgp

This is our repository for implementing Algorithm 1 from http://proceedings.mlr.press/v32/grande14.pdf for Efficient Reinforcement Learning with Gaussian Processes, along with all of our other experiments. 

experiment.py --> Implements Algorithm 1 and tests it on Experiment 1 from the paper, also contains our Gaussian Process class. 

fancy_kernel.py --> Implements the kernel from https://arxiv.org/abs/1402.5876  
fancy_kernel_experiment.py --> Runs Experiment 1 from the paper with fancy_kernel.py

torch_rbf.py --> implements radial basis kernel in pytorch for fancy_kernel.py

main.py --> Runs Algorithm 1 for modeling optimal execution of Portfolio Transactions using the Chriss/Almgren model from https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf  
ddpg_agent.py --> Implements the code for the agent's Q-function, the Gaussian Process, as well as noise sampling using an Ornstein-Uhlbeck process  
syntheticChrissAlmgren.py --> Creates a simple simulation trading environment, obtained from https://github.com/udacity/deep-reinforcement-learning/tree/master/finance

robotics_exp.py --> implements robotics experiment described in paper

gen_data.py --> creates robotics environment for robotics_exp.py



