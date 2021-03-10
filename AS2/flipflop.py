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


def flipflop():


    fitness = FlipFlop()
    problem = mlrose_hiive.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True)
    sa = mlrose_hiive.SARunner(problem=problem,
                       experiment_name='Peaks',
                       output_directory='/Users/ajinkya.bagde/Desktop/CS7641/AS2/results',
                       seed=1,
                       max_attempts=200,
                       iteration_list=[2500],
                       temperature_list=[0.05, 0.1, 0.5, 1, 10, 20, 25],
                       decay_list=[mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay, mlrose_hiive.ArithDecay])
    
    sa_stats, sa_curve = sa.run()

if __name__ == "__main__":
    flipflop()