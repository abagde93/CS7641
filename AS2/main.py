import mlrose_hiive
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import queens, peaks, tsp, flipflop

def main():
    queens.queens()
    #peaks.peaks()
    #tsp.tsp()
if __name__ == "__main__":
    main()
