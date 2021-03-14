import mlrose_hiive
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


from sklearn import preprocessing, datasets
import time
from random import randint
import warnings

def nn_impl():

    #iris_data = fetch_openml('iris')
    #X_whole, y_whole = iris_data['data'], iris_data['target']
    print("HELLO00000")

    sklearn_data = datasets.load_breast_cancer()
    x, y = sklearn_data.data, sklearn_data.target
    x = preprocessing.scale(x)

    # Split the initial data
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state=42)
    
    ### Analysis for RHC ###
    train_accuracy_scores = []
    test_accuracy_scores = []
    time_per_iteration_rhc = []

    for i in range(1,3000,50):
        print(i)
        rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='identity', 
                                     algorithm ='random_hill_climb', 
                                     bias = False, is_classifier = True, 
                                     learning_rate = 0.6, clip_max = 1,
                                     max_attempts = 1000, max_iters = i)

        start = time.time()
        rhc_nn.fit(xtrain, ytrain)
        
        
        # Train set analysis
        predictions_train = rhc_nn.predict(xtrain)
        accuracy_score_train = accuracy_score(ytrain, predictions_train)
        train_accuracy_scores.append(accuracy_score_train)

        # Test set analysis
        predictions_test = rhc_nn.predict(xtest)
        accuracy_score_test = accuracy_score(ytest, predictions_test)
        test_accuracy_scores.append(accuracy_score_test)

        time_per_iteration_rhc.append(time.time() - start)

    plt.figure()
    plt.plot(np.arange(1,3000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    plt.plot(np.arange(1,3000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations (RHC)')
    plt.legend()
    plt.savefig('testacc_iter_rhc.png')

    print("Finished RHC")

    ### Analysis for Simulated Annealing ###
    train_accuracy_scores = []
    test_accuracy_scores = []
    time_per_iteration_sa = []

    for i in range(1,3000,50):
        print(i)
        sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='simulated_annealing',
                                bias=False, is_classifier=True,
                                learning_rate = 0.6, clip_max=1,
                                max_attempts=1000, max_iters = i)

        start = time.time()
        sa_nn.fit(xtrain, ytrain)
        
        
        # Train set analysis
        predictions_train = sa_nn.predict(xtrain)
        accuracy_score_train = accuracy_score(ytrain, predictions_train)
        train_accuracy_scores.append(accuracy_score_train)

        # Test set analysis
        predictions_test = sa_nn.predict(xtest)
        accuracy_score_test = accuracy_score(ytest, predictions_test)
        test_accuracy_scores.append(accuracy_score_test)

        time_per_iteration_sa.append(time.time() - start)

    plt.figure()
    plt.plot(np.arange(1,3000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    plt.plot(np.arange(1,3000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations (SA)')
    plt.legend()
    plt.savefig('testacc_iter_SA.png')

    print("Finished SA")

    ### Analysis for Genetic Algorithms ###
    train_accuracy_scores = []
    test_accuracy_scores = []
    time_per_iteration_ga = []

    for i in range(1,3000,50):
        print(i)
        ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='genetic_alg',
                                bias=False, is_classifier=True,
                                learning_rate = 0.6, clip_max=1,
                                max_attempts=1000, max_iters = i)

        start = time.time()
        ga_nn.fit(xtrain, ytrain)
        
        
        # Train set analysis
        predictions_train = ga_nn.predict(xtrain)
        accuracy_score_train = accuracy_score(ytrain, predictions_train)
        train_accuracy_scores.append(accuracy_score_train)

        # Test set analysis
        predictions_test = ga_nn.predict(xtest)
        accuracy_score_test = accuracy_score(ytest, predictions_test)
        test_accuracy_scores.append(accuracy_score_test)

        time_per_iteration_ga.append(time.time() - start)

    plt.figure()
    plt.plot(np.arange(1,3000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    plt.plot(np.arange(1,3000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations (GA)')
    plt.legend()
    plt.savefig('testacc_iter_GA.png')

    print("Finished GA")


    ### Backpropogation (for comparison) ###
    train_accuracy_scores = []
    test_accuracy_scores = []
    time_per_iteration_bp = []
    print("backprop start")
    for i in range(1,3000,50):
        print(i)
        bp_nn = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', max_iter=i)
        # bp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
        #                         algorithm='gradient_descent',
        #                         bias=False, is_classifier=True,
        #                         learning_rate = 0.6, clip_max=1,
        #                         max_attempts=1000, max_iters = i)

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
    plt.plot(np.arange(1,3000,50),np.array(train_accuracy_scores),label='Train Accuracy')
    plt.plot(np.arange(1,3000,50),np.array(test_accuracy_scores),label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations (Backpropogation)')
    plt.legend()
    plt.savefig('testacc_iter_bp.png')

    print("Finished Backprop")

    ### Plot runtimes for above ###
    plt.figure()
    print(time_per_iteration_rhc,time_per_iteration_sa,time_per_iteration_ga,time_per_iteration_bp)
    plt.plot(np.arange(1,3000,50),np.array(time_per_iteration_rhc),label='RHC')
    plt.plot(np.arange(1,3000,50),np.array(time_per_iteration_sa),label='SA')
    plt.plot(np.arange(1,3000,50),np.array(time_per_iteration_ga),label='GA')
    plt.plot(np.arange(1,3000,50),np.array(time_per_iteration_bp),label='BP')
    plt.xlabel('Iterations')
    plt.ylabel('Training Time')
    plt.title('Training Time vs Iterations')
    plt.legend()
    plt.savefig('time_vs_iter.png')

    # #### Hyperparameter Tuning - RHC ####
    # ## Adjusting the number of random restarts ##
    # train_accuracy_scores = []
    # test_accuracy_scores = []

    # for i in range(0,500,25):
    #     print(i)
    #     rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes = [2], activation ='identity', 
    #                                  algorithm ='random_hill_climb', 
    #                                  bias = False, is_classifier = True, 
    #                                  learning_rate = 0.6, clip_max = 1,
    #                                  max_attempts = 1000, restarts = i)

    #     rhc_nn.fit(xtrain, ytrain)
        
        
    #     # Train set analysis
    #     predictions_train = rhc_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = rhc_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)

    # plt.figure()
    # plt.plot(np.arange(0,500,25),np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(np.arange(0,500,25),np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('Restarts')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Number of Restarts (RHC)')
    # plt.legend()
    # plt.savefig('rhc_restarts.png')

    # print("Finished RHC HP Tuning")

    # #### Hyperparameter Tuning - SA ####
    # ## Adjusting the type of scheduling ##
    # train_accuracy_scores = []
    # test_accuracy_scores = []

    # # Referending sectiion 2.2 'Decay Schedules' here:
    # # https://readthedocs.org/projects/mlrose/downloads/pdf/stable/

    # schedule_types = [mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay(), mlrose_hiive.GeomDecay()]

    # for st in schedule_types:
    #     print(st)
    #     sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='identity',
    #                             algorithm='simulated_annealing',
    #                             bias=False, is_classifier=True,
    #                             learning_rate = 0.6, clip_max=1,
    #                             max_attempts=1000, schedule = st)

    #     sa_nn.fit(xtrain, ytrain)
        
    #     # Train set analysis
    #     predictions_train = sa_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = sa_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)

    # plt.figure()
    # plt.plot(['ExpDecay','ArithDecay','GeomDecay'],np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(['ExpDecay','ArithDecay','GeomDecay'],np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('Schedule Type')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Schedule Type (SA)')
    # plt.legend()
    # plt.savefig('sa_schedule_type.png')

    # print("Finished SA HP Tuning")

    # #### Hyperparameter Tuning - GA ####

    # ## Adjusting the amount of mutation
    # ## Used api as referenced in https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
    # train_accuracy_scores = []
    # test_accuracy_scores = []

    # mutation_prob_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for i in mutation_prob_array:
    #     ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='relu',
    #                             algorithm='genetic_alg',
    #                             bias=False, is_classifier=True,
    #                             learning_rate = 0.6, clip_max=1,
    #                             max_attempts=1000, mutation_prob = i)

    #     ga_nn.fit(xtrain, ytrain)
        
    #     # Train set analysis
    #     predictions_train = ga_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = ga_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)


    # plt.figure()
    # plt.plot(mutation_prob_array,np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(mutation_prob_array,np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('mutation_prob')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Iterations (GA - mutation_prob experimentation)')
    # plt.legend()
    # plt.savefig('ga_mutation.png')

    # print("Finished GA mutation experimentation")

    # ## Adjusting the population size
    # ## Used api as referenced in https://readthedocs.org/projects/mlrose/downloads/pdf/stable/
    # train_accuracy_scores = []
    # test_accuracy_scores = []

    # pop_size_array = [100,200,300,400,500]
    # for i in pop_size_array:
    #     ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[2], activation='relu',
    #                             algorithm='genetic_alg',
    #                             bias=False, is_classifier=True,
    #                             learning_rate = 0.6, clip_max=1,
    #                             max_attempts=1000, pop_size = i)

    #     ga_nn.fit(xtrain, ytrain)
        
    #     # Train set analysis
    #     predictions_train = ga_nn.predict(xtrain)
    #     accuracy_score_train = accuracy_score(ytrain, predictions_train)
    #     train_accuracy_scores.append(accuracy_score_train)

    #     # Test set analysis
    #     predictions_test = ga_nn.predict(xtest)
    #     accuracy_score_test = accuracy_score(ytest, predictions_test)
    #     test_accuracy_scores.append(accuracy_score_test)


    # plt.figure()
    # plt.plot(pop_size_array,np.array(train_accuracy_scores),label='Train Accuracy')
    # plt.plot(pop_size_array,np.array(test_accuracy_scores),label='Test Accuracy')
    # plt.xlabel('pop_size')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs. Iterations (GA - pop_size experimentation)')
    # plt.legend()
    # plt.savefig('ga_popsize.png')

    # print("Finished GA pop_size experimentation")





    

        


nn_impl()