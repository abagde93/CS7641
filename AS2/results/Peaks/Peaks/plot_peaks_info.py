import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import pandas as pd
import matplotlib.pyplot as plt

from mlrose_hiive.algorithms.decay import ExpDecay
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks

df_ga = pd.read_csv('ga__Peaks__curves_df.csv')
df_mimic = pd.read_csv('mimic__Peaks__curves_df.csv')
df_sa = pd.read_csv('sa__Peaks__curves_df.csv')
df_rhc = pd.read_csv('rhc__Peaks__curves_df.csv')

def plot_fitness_iteration(name, curve, title, x="Iteration", y="Fitness"):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    length = len(curve)
    plt.plot(range(length), curve, label=y, lw=2)
    plt.legend(loc="best")
    plt.savefig(name)

## MIMIC analysis

max_attempts = 10
max_iters = 25
keep_pct=0.25
pop_size = 1000
global eval_count
eval_count = 0
best_state, best_fitness, mimic_curve = mlrose_hiive.mimic(prob,
                                             max_attempts=max_attempts,
                                             max_iters=max_iters,
                                             random_state=random_state,
                                             pop_size=pop_size,
                                             keep_pct=keep_pct,
                                             curve=True)
print("MIMIC - Total Function Evaluations:", eval_count)
plot_fitness_iteration('fitness_iterations.png',mimic_curve,
                       "Queens - Mimic: Population Size: {}, Keep Percent: {}".format(pop_size, keep_pct))

