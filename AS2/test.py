import mlrose_hiive
from mlrose_hiive.algorithms.decay import ExpDecay
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
def queens_max(state):
    # Define alternative N-Queens fitness function for maximization problem
    # Initialize counter
    fitness = 0
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness += 1
    return fitness
def queens():
    '''
    Taken basically verbatim from https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb
    :return:
    '''
    # Initialize fitness function object using pre-defined class
    # fitness = mlrose_hiive.Queens()
    fitness = mlrose_hiive.CustomFitness(queens_max)
    problem = mlrose_hiive.QueensOpt(length=8, fitness_fn=fitness, maximize=True)
    sa = mlrose_hiive.SARunner(problem=problem,
                       experiment_name='Queens',
                       output_directory='results',
                       seed=1,
                       max_attempts=200,
                       iteration_list=[2500],
                       temperature_list=[0.05, 0.1, 0.5, 1, 10, 20, 25],
                       decay_list=[mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay, mlrose_hiive.ArithDecay])
    sa_stats, sa_curve = sa.run()