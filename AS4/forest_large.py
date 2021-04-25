import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

import mdptoolbox, mdptoolbox.example

from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning, PolicyIterationModified
from hiive.mdptoolbox.example import forest

plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS4/forest_large/'

print('POLICY ITERATION WITH FOREST MANAGEMENT')
P, R = mdptoolbox.example.forest(S=2000)
value_f = [0]*10
policy = [0]*10
iters = [0]*10
time_array = [0]*10
gamma_arr = [0] * 10
f = open(plot_path + "policyiteration_optimalpolicy.txt", "a")
for i in range(0,10):
    pi = mdptoolbox.mdp.PolicyIteration(P, R, (i+0.1)/10)
    pi.run()
    gamma_arr[i]=(i+0.1)/10
    value_f[i] = np.mean(pi.V)
    policy[i] = pi.policy
    iters[i] = pi.iter
    time_array[i] = pi.time
    f.write("Optimal Policy for gamma: " + str(i+.1/10))
    f.write(str(pi.policy))
f.close()

plt.figure()
plt.plot(gamma_arr, time_array)
plt.xlabel('Gammas')
plt.title('Forest Management - Policy Iteration - Execution Time Analysis')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(plot_path + 'policy_execution_time_analysis.png')

plt.figure()
plt.plot(gamma_arr,value_f)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Forest Management - Policy Iteration - Reward Analysis')
plt.grid()
plt.savefig(plot_path + 'policy_reward_analysis.png')

plt.figure()
plt.plot(gamma_arr,iters)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Forest Management - Policy Iteration - Convergence Analysis')
plt.grid()
plt.savefig(plot_path + 'policy_convergence_analysis.png')


print('VALUE ITERATION WITH FOREST MANAGEMENT')
P, R = mdptoolbox.example.forest(S=2000)
value_f = [0]*10
policy = [0]*10
iters = [0]*10
time_array = [0]*10
gamma_arr = [0] * 10
f = open(plot_path + "valueiteration_optimalpolicy.txt", "a")
for i in range(0,10):
    pi = mdptoolbox.mdp.ValueIteration(P, R, (i+0.1)/10)
    pi.run()
    gamma_arr[i]=(i+0.5)/10
    value_f[i] = np.mean(pi.V)
    policy[i] = pi.policy
    iters[i] = pi.iter
    time_array[i] = pi.time
    f.write("Optimal Policy for gamma: " + str(i+.1/10))
    f.write(str(pi.policy))
f.close()

plt.figure()
plt.plot(gamma_arr, time_array)
plt.xlabel('Gammas')
plt.title('Forest Management - Value Iteration - Execution Time Analysis')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(plot_path + 'value_execution_time_analysis.png')

plt.figure()
plt.plot(gamma_arr,value_f)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Forest Management - Value Iteration - Reward Analysis')
plt.grid()
plt.savefig(plot_path + 'value_reward_analysis.png')

plt.figure()
plt.plot(gamma_arr,iters)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Forest Management - Value Iteration - Convergence Analysis')
plt.grid()
plt.savefig(plot_path + 'value_convergence_analysis.png')

print('Q LEARNING WITH FOREST MANAGEMENT')

P, R = mdptoolbox.example.forest(S=2000,p=0.01)
value_f = []
policy = []
iters = []
time_array = []
Q_table = []
rew_array = []

# Plots for variable iterations

niters = [10000, 25000, 50000, 100000, 250000, 500000]
for niter in niters:
    print("doing iteration ", niter)
    ql = QLearning(P, R, 0.95, n_iter=niter)
    ql.run()
    time = ql.time
    maxV = np.amax(ql.V)
    rew_array.append(maxV)
    Q_table.append(ql.Q)
    policy.append(ql.policy)
    time_array.append(time)

plt.figure()
plt.plot(niters, rew_array, label='epsilon=0.95')
plt.title('Forest QLearning: Iteration vs average rewards')
plt.savefig(plot_path + 'qlearning_iteration_rewards_analysis.png')

plt.figure()
plt.plot(niters, time_array, label='epsilon=0.95')
plt.title('Forest QLearning: Iteration vs Time')
plt.savefig(plot_path + 'qlearning_iteration_time_analysis.png')

Plots for variable gammas

gammas = np.arange(0.1, 0.99, 0.04)
for gamma in gammas:
    print("doing gamma ", gamma)
    ql = QLearning(P, R, gamma, n_iter=10000, epsilon=.95)
    ql.run()
    time = ql.time
    maxV = np.amax(ql.V)
    rew_array.append(maxV)
    Q_table.append(ql.Q)
    policy.append(ql.policy)
    time_array.append(time)

plt.figure()
plt.plot(gammas, rew_array, label='epsilon=0.95')
plt.title('Forest QLearning: Gamma vs average rewards')
plt.savefig(plot_path + 'qlearning_gamma_rewards_analysis.png')

plt.figure()
plt.plot(gammas, time_array, label='epsilon=0.95')
plt.title('Forest QLearning: Gamma vs Time')
plt.savefig(plot_path + 'qlearning_gamma_time_analysis.png')

# Plots for variable epsilons
epsilons = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
for epsilon_val in epsilons:
    print("doing epsilon ", epsilon_val)
    ql = QLearning(P, R, epsilon=epsilon_val, n_iter=10000, gamma=0.9)
    ql.run()
    time = ql.time
    maxV = np.amax(ql.V)
    rew_array.append(maxV)
    Q_table.append(ql.Q)
    policy.append(ql.policy)
    time_array.append(time)

plt.figure()
plt.plot(epsilons, rew_array, label='n_iter=100000')
plt.title('Forest QLearning: Epsilon vs average rewards')
plt.savefig(plot_path + 'qlearning_epsilon_rewards_analysis.png')

plt.figure()
plt.plot(epsilons, time_array, label='n_iter=100000')
plt.title('Forest QLearning: Epsilon vs Time')
plt.savefig(plot_path + 'qlearning_epsilon_time_analysis.png')
