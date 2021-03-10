import mlrose_hiive
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import queens, peaks, tsp, flipflop
import matplotlib.pyplot as plt


from sklearn import preprocessing, datasets
import time
from random import randint
import warnings

def nn():

    #iris_data = fetch_openml('iris')
    #X_whole, y_whole = iris_data['data'], iris_data['target']


    sklearn_data = datasets.load_breast_cancer()
    x, y = sklearn_data.data, sklearn_data.target
    x = preprocessing.scale(x)

    # Split the initial data
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state=42)
    
    # ### Analysis for RHC ###
    # train_accuracy_scores = []
    # test_accuracy_scores = []
    # time_per_iteration_rhc = []

    # for i in range(100,1000,50):
    #     print(i)
    #     rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='identity', 
    #                                  algorithm ='random_hill_climb', 
    #                                  bias = False, is_classifier = True, 
    #                                  learning_rate = 0.6, clip_max = 1,
    #                                  max_attempts = 1000, max_iters = i)

    #     start = time.time()
    #     rhc_nn.fit(xtrain, ytrain)
        
        
    #     # Train set analysis
    #     predictions_train = rhc_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = rhc_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)

    #     time_per_iteration_rhc.append(time.time() - start)

    # plt.figure()
    # plt.plot(np.arange(100,1000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(np.arange(100,1000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Iterations (RHC)')
    # plt.legend()
    # plt.savefig('testacc_iter_rhc.png')

    # print("Finished RHC")

    # ### Analysis for Simulated Annealing ###
    # train_accuracy_scores = []
    # test_accuracy_scores = []
    # time_per_iteration_sa = []

    # for i in range(100,1000,50):
    #     print(i)
    #     sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
    #                             algorithm='simulated_annealing',
    #                             bias=False, is_classifier=True,
    #                             learning_rate = 0.6, clip_max=1,
    #                             max_attempts=1000, max_iters = i)

    #     start = time.time()
    #     sa_nn.fit(xtrain, ytrain)
        
        
    #     # Train set analysis
    #     predictions_train = sa_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = sa_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)

    #     time_per_iteration_sa.append(time.time() - start)

    # plt.figure()
    # plt.plot(np.arange(100,1000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(np.arange(100,1000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Iterations (SA)')
    # plt.legend()
    # plt.savefig('testacc_iter_SA.png')

    # print("Finished SA")

    # ### Analysis for Genetic Algorithms ###
    # train_accuracy_scores = []
    # test_accuracy_scores = []
    # time_per_iteration_ga = []

    # for i in range(100,1000,50):
    #     print(i)
    #     ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
    #                             algorithm='genetic_alg',
    #                             bias=False, is_classifier=True,
    #                             learning_rate = 0.6, clip_max=1,
    #                             max_attempts=1000, max_iters = i)

    #     start = time.time()
    #     ga_nn.fit(xtrain, ytrain)
        
        
    #     # Train set analysis
    #     predictions_train = ga_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = ga_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)

    #     time_per_iteration_ga.append(time.time() - start)

    # plt.figure()
    # plt.plot(np.arange(100,1000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(np.arange(100,1000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Iterations (GA)')
    # plt.legend()
    # plt.savefig('testacc_iter_GA.png')

    # print("Finished GA")

    # ### Plot runtimes for above ###
    # plt.figure()
    # plt.plot(np.arange(100,1000,50),np.array(time_per_iteration_rhc),label='RHC')
    # plt.plot(np.arange(100,1000,50),np.array(time_per_iteration_sa),label='SA')
    # plt.plot(np.arange(100,1000,50),np.array(time_per_iteration_ga),label='GA')
    # plt.xlabel('Iterations')
    # plt.ylabel('Training Time')
    # plt.title('Training Time vs Iterations')
    # plt.legend()
    # plt.savefig('time_vs_iter.png')

    ### Backpropogation (for comparison) ###
    train_accuracy_scores = []
    test_accuracy_scores = []
    time_per_iteration_bp = []

    for i in range(100,5000,50):
        print(i)
        bp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='sigmoid',
                                algorithm='gradient_descent',
                                bias=False, is_classifier=True,
                                learning_rate = 0.6, clip_max=1,
                                max_attempts=1000, max_iters = i)

        start = time.time()
        bp_nn.fit(xtrain, ytrain)
        
        
        # Train set analysis
        predictions_train = bp_nn.predict(xtrain)
        accuracy_score_train = accuracy_score(ytrain, predictions_train)
        train_accuracy_scores.append(accuracy_score_train)

        # Test set analysis
        predictions_test = bp_nn.predict(xtest)
        accuracy_score_test = accuracy_score(ytest, predictions_test)
        test_accuracy_scores.append(accuracy_score_test)

        time_per_iteration_bp.append(time.time() - start)

    plt.figure()
    plt.plot(np.arange(100,5000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    plt.plot(np.arange(100,5000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations (Backpropogation)')
    plt.legend()
    plt.savefig('testacc_iter_bp.png')

    print("Finished Backprop")



    

        


nn()