import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning, PolicyIterationModified


##### FUNCTIONS USED #####


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(env, v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            k=i+1
            break
        policy = new_policy
    return policy, k

def value_iteration(env, gamma ):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    desc = env.unwrapped.desc
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            k=i+1
            break
    return v,k

def plot_policy_map(title, policy, map_desc, color_map, direction_map):

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'

    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)
     
            text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')
 
    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()

    return plt

def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'black',
		b'G': 'gold',
	}

def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}



##### EXPERIMENT 1 (frozenlake - small) #####
plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS4/frozenlake_small/'

env = gym.make('FrozenLake-v0')
env = env.unwrapped
desc = env.unwrapped.desc


list_scores, time_array, gamma_array, iters, best_vals = [0]*100, [0]*100, [0]*100, [0]*100, [0]*100

print('Frozen Lake (Small) - Policy Iteration')
for i in range(0,100):

    # Get Policy Iteration Metrics
    st=time.time()
    best_policy,k = policy_iteration(env, gamma =i/100)
    scores = evaluate_policy(env, best_policy, gamma = i/100)
    end=time.time()

    # Store Policy Iteraion Metrics
    gamma_array[i]=i/100
    list_scores[i]=np.mean(scores)
    iters[i] = k
    time_array[i]=end-st




plt.plot(gamma_array, time_array)
plt.xlabel('Gammas')
plt.title('Frozen Lake - Policy Iteration - Execution Time Analysis')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(plot_path + 'policy_execution_time_analysis.png')

plt.plot(gamma_array,list_scores)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Frozen Lake - Policy Iteration - Reward Analysis')
plt.grid()
plt.savefig(plot_path + 'policy_reward_analysis.png')

plt.plot(gamma_array,iters)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Frozen Lake - Policy Iteration - Convergence Analysis')
plt.grid()
plt.savefig(plot_path + 'policy_convergence_analysis.png')


print('Frozen Lake (Small) - Value Iteration')
for i in range(0,100):

    # Get Value Iteration Metrics & plot policy map
    st=time.time()
    best_value,k = value_iteration(env, gamma =i/100)
    policy = extract_policy(env,best_value, gamma = i/100)
    policy_score = evaluate_policy(env, policy, gamma = i/100)
    gamma = i/100
    plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),policy.reshape(4,4),desc,colors_lake(),directions_lake())
    plt.savefig(plot_path + '/maps/' + 'iteration_' + str(i) + "_gamma_value_" + str(gamma) + '.png')
    end=time.time()


    # Store Value Iteration Metrics
    gamma_array[i]=i/100
    iters[i]=k
    best_vals[i] = best_value
    list_scores[i]=np.mean(policy_score)
    time_array[i]=end-st

plt.figure()
plt.plot(gamma_array, time_array)
plt.xlabel('Gammas')
plt.title('Frozen Lake - Value Iteration - Execution Time Analysis')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.savefig(plot_path + 'value_execution_time_analysis.png')

plt.figure()
plt.plot(gamma_array,list_scores)
plt.xlabel('Gammas')
plt.ylabel('Average Rewards')
plt.title('Frozen Lake - Value Iteration - Reward Analysis')
plt.grid()
plt.savefig(plot_path + 'value_reward_analysis.png')

plt.figure()
plt.plot(gamma_array,iters)
plt.xlabel('Gammas')
plt.ylabel('Iterations to Converge')
plt.title('Frozen Lake - Value Iteration - Convergence Analysis')
plt.grid()
plt.savefig(plot_path + 'value_convergence_analysis.png')

plt.figure()
plt.plot(gamma_array,best_vals)
plt.xlabel('Gammas')
plt.ylabel('Optimal Value')
plt.title('Frozen Lake - Value Iteration - Best Value Analysis')
plt.grid()
plt.savefig(plot_path + 'value_bestvalue_analysis.png')


print('Frozen Lake (Small) - Q-Learning')

states = env.observation_space.n
actions = env.action_space.n

P = np.zeros((actions, states, states))
R = np.zeros((states, actions))

for s in env.P:
    for a in env.P[s]:
        for decision in env.P[s][a]:
            P[a][s][decision[1]] += decision[0]
            R[s][a] += decision[2]


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
plt.title('Frozenlake QLearning: Iteration vs average rewards')
plt.savefig(plot_path + 'qlearning_iteration_rewards_analysis.png')

plt.figure()
plt.plot(niters, time_array, label='epsilon=0.95')
plt.title('Frozenlake QLearning: Iteration vs Time')
plt.savefig(plot_path + 'qlearning_iteration_time_analysis.png')

# Plots for variable gammas

gammas = np.arange(0.1, 0.99, 0.04)
for gamma in gammas:
    print("doing gamma ", gamma)
    ql = QLearning(P, R, gamma, n_iter=10000)
    ql.run()
    time = ql.time
    maxV = np.amax(ql.V)
    rew_array.append(maxV)
    Q_table.append(ql.Q)
    policy.append(ql.policy)
    time_array.append(time)

plt.figure()
plt.plot(gammas, rew_array, label='epsilon=0.95')
plt.title('Frozenlake QLearning: Gamma vs average rewards')
plt.savefig(plot_path + 'qlearning_gamma_rewards_analysis.png')

plt.figure()
plt.plot(gammas, time_array, label='epsilon=0.95')
plt.title('Frozenlake QLearning: Gamma vs Time')
plt.savefig(plot_path + 'qlearning_gamma_time_analysis.png')

# Plots for variable epsilons
epsilons = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
for epsilon_val in epsilons:
    print("doing epsilon ", epsilon_val)
    ql = QLearning(P, R, epsilon=epsilon_val, n_iter=500000, gamma=0.9)
    ql.run()
    time = ql.time
    maxV = np.amax(ql.V)
    rew_array.append(maxV)
    Q_table.append(ql.Q)
    policy.append(ql.policy)
    time_array.append(time)

plt.figure()
plt.plot(epsilons, rew_array, label='n_iter=500000')
plt.title('Frozenlake QLearning: Epsilon vs average rewards')
plt.savefig(plot_path + 'qlearning_epsilon_rewards_analysis.png')

plt.figure()
plt.plot(epsilons, time_array, label='n_iter=500000')
plt.title('Frozenlake QLearning: Epsilon vs Time')
plt.savefig(plot_path + 'qlearning_epsilon_time_analysis.png')




