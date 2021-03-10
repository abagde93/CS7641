import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive

#import mlrose_hiive
from mlrose_hiive.algorithms.decay import ExpDecay
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks


def tsp():

    # Define list of inter-distances between each pair of the following cities (in order from 0 to 9):
    # Rome, Florence, Barcelona, Paris, London, Amsterdam, Berlin, Prague, Budapest, Venice
    distances = [(0, 1, 0.274), (0, 2, 1.367), (1, 2, 1.091), (0, 3, 1.422), (1, 3, 1.153), (2, 3, 1.038),
                (0, 4, 1.870), (1, 4, 1.602), (2, 4, 1.495), (3, 4, 0.475), (0, 5, 1.652), (1, 5, 1.381),
                (2, 5, 1.537), (3, 5, 0.515), (4, 5, 0.539), (0, 6, 1.504), (1, 6, 1.324), (2, 6, 1.862),
                (3, 6, 1.060), (4, 6, 1.097), (5, 6, 0.664), (0, 7, 1.301), (1, 7, 1.031), (2, 7, 1.712),
                (3, 7, 1.031), (4, 7, 1.261), (5, 7, 0.893), (6, 7, 0.350), (0, 8, 1.219), (1, 8, 0.948),
                (2, 8, 1.923), (3, 8, 1.484), (4, 8, 1.723), (5, 8, 1.396), (6, 8, 0.872), (7, 8, 0.526),
                (0, 9, 0.529), (1, 9, 0.258), (2, 9, 1.233), (3, 9, 1.137), (4, 9, 1.560), (5, 9, 1.343),
                (6, 9, 1.131), (7, 9, 0.816), (8, 9, 0.704)]

    fitness = mlrose_hiive.TravellingSales(distances=distances)
    problem = mlrose_hiive.TSPOpt(length=8, fitness_fn=fitness, maximize=True)


    exp_name = 'tsp'
    out_dir = '/Users/ajinkya.bagde/Desktop/CS7641/AS2/results/tsp'
    random_state = 100

    # Simulated Annealing
    sa = mlrose_hiive.SARunner(problem=problem,
                       experiment_name=exp_name,
                       output_directory=out_dir,
                       seed=1,
                       max_attempts=200,
                       iteration_list=[2500],
                       temperature_list=[0.05, 0.1, 0.5, 1, 10, 20, 25],
                       decay_list=[mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay, mlrose_hiive.ArithDecay])
    
    sa_stats, sa_curve = sa.run()

    # MIMIC
    mim = mlrose_hiive.MIMICRunner(problem=problem,
                       experiment_name=exp_name,
                       output_directory=out_dir,
                       seed=random_state,
                       population_sizes=[1000, 2000, 3000],
                       keep_percent_list=[0.05, 0.1, 0.15, 0.20],
                       iteration_list=[25],
                       max_attempts=100,
                       use_fast_mimic=True)
    mim_stats, mim_curve = mim.run()

    # Genetic Algorithm
    ga = mlrose_hiive.GARunner(problem=problem,
                   experiment_name=exp_name,
                   output_directory=out_dir,
                   seed=random_state,
                   max_attempts=20,
                   iteration_list=[25],
                   population_sizes=[500, 1000, 2000, 3000],
                   mutation_rates=[0.1, 0.25, 0.5, 0.75])
    ga_stats, ga_curve = ga.run()

    # Randomized Hill Climbing
    rhc = mlrose_hiive.RHCRunner(problem=problem,
                    experiment_name=exp_name,
                    output_directory=out_dir,
                    seed=random_state,
                    max_attempts=200,
                    iteration_list=[2500],
                    restart_list=[10])
    rhc_stats, rhc_curve = rhc.run()


if __name__ == "__main__":
    tsp()