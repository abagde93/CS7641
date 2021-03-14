import time
import mlrose_hiive as mlrh
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 7]

random_state = 2020
np.random.seed(1)

def plot_fitness_iteration(name, curve, title, x="Iteration", y="Fitness"):
    plt.figure()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    length = len(curve)
    plt.plot(range(length), curve, label=y, lw=2)
    plt.legend(loc="best")
    plt.savefig(name)

# Problem definition
length = 35
global eval_count
eval_count = 0
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    global eval_count
    eval_count += 1
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

# Initialize custom fitness function object
q_fitness = mlrh.CustomFitness(queens_max)
# q_fitness = mlrose.Queens()
prob = mlrh.QueensOpt(length=length,
                      fitness_fn=q_fitness,
                      maximize=True)
experiment_name = "queen_prob"
output_directory = "queen"

# SA
sa = mlrh.SARunner(problem=prob,
                   experiment_name=experiment_name,
                   output_directory=output_directory,
                   seed=random_state,
                   max_attempts=200,
                   iteration_list=[2000],
                   temperature_list=[0.01, 0.1, 1, 10, 100, 1000],
                   decay_list=[mlrh.GeomDecay, mlrh.ExpDecay, mlrh.ArithDecay])
sa_stats, sa_curve = sa.run()


columns = ['Time', 'Fitness', 'Temperature', 'schedule_type']
df=pd.read_csv("./queen/queen_prob/sa__queen_prob__run_stats_df.csv")
print(df[columns].sort_values(by=['Fitness'], ascending=False))


max_attempts = 200
max_iters = 2000
init_temp = 0.01
schedule = mlrh.ExpDecay(init_temp)
eval_count = 0
best_state, best_fitness, sa_curve = mlrh.simulated_annealing(prob,
                                                             max_attempts=max_attempts,
                                                             max_iters=max_iters,
                                                             random_state=random_state,
                                                             schedule=schedule,
                                                             curve=True)
print("Simulated Annealing - Total Function Evaluations:", eval_count)
plot_fitness_iteration('fitness_iteration_sa_queens.png', sa_curve,
                       "Queens - Simulated Annealing: schedule: {}, init_temp: {}".format(schedule.__class__.__name__, init_temp))

# GA
ga = mlrh.GARunner(problem=prob,
                   experiment_name=experiment_name,
                   output_directory=output_directory,
                   seed=random_state,
                   max_attempts=20,
                   iteration_list=[100],
                   population_sizes=[10, 100, 200, 300],
                   mutation_rates=[0.1, 0.25, 0.5, 0.75, 1.0])
ga_stats, ga_curve = ga.run()



columns = ['Time', 'Fitness', 'Population Size', 'Mutation Rate']
df=pd.read_csv("./queen/queen_prob/ga__queen_prob__run_stats_df.csv")
print(df[columns].sort_values(by=['Fitness'], ascending=False))


max_attempts = 20
max_iters = 25
mutation_prob=0.5
pop_size = 10
eval_count = 0
best_state, best_fitness, gen_curve = mlrh.genetic_alg(prob,
                                                   max_attempts=max_attempts,
                                                   max_iters=max_iters,
                                                   random_state=random_state,
                                                   pop_size=pop_size,
                                                   mutation_prob=mutation_prob,
                                                   curve=True)
print("Genetic Alg - Total Function Evaluations:", eval_count)
plot_fitness_iteration('fitness_iteration_ga_queens.png',gen_curve,
                       "Queens - Genetic Alg: mutation_prob: {}, pop_size: {}".format(mutation_prob, pop_size))


# MIMIC
mim = mlrh.MIMICRunner(problem=prob,
                       experiment_name=experiment_name,
                       output_directory=output_directory,
                       seed=random_state,
                       population_sizes=[50, 100, 200],
                       keep_percent_list=[0.1, 0.25, 0.5, 0.75],
                       iteration_list=[50],
                       use_fast_mimic=True)
mim_stats, mim_curve = mim.run()

columns = ['Time', 'Fitness', 'Population Size', 'Keep Percent']
df=pd.read_csv("./queen/queen_prob/mimic__queen_prob__run_stats_df.csv")
print(df[columns].sort_values(by=['Fitness'], ascending=False))

max_attempts = 10
max_iters = 25
keep_pct=0.25
pop_size = 200
eval_count = 0
best_state, best_fitness, mimic_curve = mlrh.mimic(prob,
                                             max_attempts=max_attempts,
                                             max_iters=max_iters,
                                             random_state=random_state,
                                             pop_size=pop_size,
                                             keep_pct=keep_pct,
                                             curve=True)
print("MIMIC - Total Function Evaluations:", eval_count)
plot_fitness_iteration('fitness_iteration_mimic_queens.png',mimic_curve,
                       "Queens - Mimic: Population Size: {}, Keep Percent: {}".format(pop_size, keep_pct))

# RHC

rhc = mlrh.RHCRunner(problem=prob,
                    experiment_name=experiment_name,
                    output_directory=output_directory,
                    seed=random_state,
                    max_attempts=200,
                    iteration_list=[2500],
                    restart_list=[20])
rhc_stats, rhc_curve = rhc.run()

columns = ['Time', 'Fitness', 'Restarts', 'current_restart']
df=pd.read_csv("./queen/queen_prob/rhc__queen_prob__run_stats_df.csv")
print(df[columns].sort_values(by=['Fitness'], ascending=False))

max_attempts = 500
max_iters = 2500
restarts = 20
eval_count = 0
best_state, best_fitness, rhc_curve = mlrh.random_hill_climb(prob,
                                                         max_attempts=max_attempts,
                                                         max_iters=max_iters,
                                                         random_state=random_state,
                                                         restarts=restarts,
                                                         curve=True)
print("Randomized Hill Climbing - Total Function Evaluations:", eval_count)
plot_fitness_iteration('fitness_iteration_rhc_queens.png',rhc_curve,
                       "Queens - Randomized Hill Climbing: restarts: {}".format(restarts))


# Fitness vs iterations (combined)

all_curves = {"MIMIC": mimic_curve, "Genetic Alg": gen_curve, "Simulated Annealing": sa_curve, "Random Hill Climb": rhc_curve}

np.array([len(x) for x in all_curves]).max()

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("N-Queens")

for name, curve in all_curves.items():
    plt.plot(range(len(curve)), curve, label=name, lw=2)

    plt.legend(loc="best")
plt.savefig('queens_combined.png')

