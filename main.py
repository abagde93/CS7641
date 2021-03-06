from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import numpy as np

import DTLearner as dt
import NNLearner as nn
import SVMLearner as svml
import KNNLearner as knn
import BoostingLearner as boost

from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier


def get_data():
    # Fetch and split Data here
    mnist_data = fetch_openml('mnist_784')
    #print(mnist_data.keys())
    X_whole, y_whole = mnist_data['data'], mnist_data['target']

    # Take a subset of the data (10%)
    X = X_whole[0::20]
    y = y_whole[0::20]

    # Lets validate this data (we want to see that the 10% subset is still representative of the actual data)
    fig, ax = plt.subplots(2)
    whole_cats, whole_counts = np.unique(y_whole, return_counts=True)
    subset_cats, subset_counts = np.unique(y, return_counts=True)
    ax[0].bar(whole_cats, whole_counts, label = 'Class Distribution (Entire Set)')
    ax[1].bar(subset_cats, subset_counts, label = 'Class Distribution (Subset)')
    plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/class_distribution.png')

    return X_whole, y_whole, X, y

def test_DT(X_whole, y_whole, X, y):

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

    start=datetime.now()

    ### DTLearner Implementation ###
    dtlearner = dt.DTLearner(leaf_size=1, n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

    ########## Initial Learning Curves for Different Pruning Values ##########

    # ccp_alpha = 0.0
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0.png')

    # ccp_alpha = 0.0002
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0002)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0002)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0002.png')

    # ccp_alpha = 0.0004
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0004)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0004)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0004.png')

    # ccp_alpha = 0.0006
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0006)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0006)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0006.png')

    # ccp_alpha = 0.0008
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0008)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0008)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0008.png')

    # ccp_alpha = 0.0010
    # Initial Fit
    initial_classifier = DecisionTreeClassifier(ccp_alpha=0.0010)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Decision Trees - ccp_alpha=0.0010)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve_initial_ccpa_0010.png')

    # Get a list of possible decision trees and their respective alphas
    flag = 0
    clfs, alphas = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the alpha with highest accuracy
    depths = "NA"
    min_samples_leafs = "NA"
    min_samples_splits = "NA"
    dt_choice_alpha_based = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,min_samples_leafs,min_samples_splits,flag)

    # Get a list of possible decision trees and their respective depths
    flag = 1
    clfs, depths = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the depth with highest accuracy
    alphas = "NA"
    min_samples_leafs = "NA"
    min_samples_splits = "NA"
    dt_choice_depth_based = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,min_samples_leafs,min_samples_splits,flag)

    # Get a list of possible decision trees and their respective min_samples_leaf
    flag = 2
    clfs, min_samples_leafs = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the min_samples_leaf with highest accuracy
    alphas = "NA"
    depths = "NA"
    min_samples_splits = "NA"
    dt_choice_msl_based = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,min_samples_leafs,min_samples_splits,flag)

    # Get a list of possible decision trees and their respective min_samples_split
    flag = 3
    clfs, min_samples_splits = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the min_samples_split with highest accuracy
    alphas = "NA"
    depths = "NA"
    min_samples_leafs = "NA"
    dt_choice_mss_based = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,min_samples_leafs,min_samples_splits,flag)

    # Now that we have the decision tree, time for tuning hyperparameters
    # Make a new classifier for this
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(xtrain_val, ytrain_val)
    best_params = dtlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
    print("Best params are: ", best_params)

    # Now do one more fit based on best params above
    final_classifier = DecisionTreeClassifier(random_state=0,ccp_alpha=best_params['ccp_alpha'],criterion=best_params['criterion'], max_depth=best_params['max_depth'],min_samples_leaf=best_params['min_samples_leaf'],min_samples_split=best_params['min_samples_split'])
    final_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Learning Curves (Decision Trees)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = final_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/dt_learningcurve.png')

    # Now time for final accuracy score for test set
    dtlearner.final_test(final_classifier,xtest,ytest)

    print(datetime.now()-start)

def test_NN(X_whole, y_whole, X, y):
    

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

    start=datetime.now()

    ### NNLearner Implementation ###
    nnlearner = nn.NNLearner(n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

    # Initial Fit
    initial_classifier = MLPClassifier()
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Neural Nets)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/nn_learningcurve_initial.png')

    # Get a list of possible neural nets and their respective activation_types
    flag = 0
    clfs, activation_types = nnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the nn that is correlated to the activation_type with highest accuracy
    momentum_values = "NA"
    learning_rates = "NA"
    hidden_layer_sizes_types = "NA"
    nn_choice_activation_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

    # Get a list of possible neural nets and their respective momentum values
    flag = 1
    clfs, momentum_values = nnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the nn that is correlated to the momentum with highest accuracy
    activation_types = "NA"
    learning_rates = "NA"
    hidden_layer_sizes_types = "NA"
    nn_choice_momentum_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

    # Get a list of possible neural nets and their respective learning rates
    flag = 2
    clfs, learning_rates = nnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the nn that is correlated to the learning rate with highest accuracy
    activation_types = "NA"
    momentum_values = "NA"
    hidden_layer_sizes_types = "NA"
    nn_choice_lr_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

    # Get a list of possible neural nets and their respective hidden_layer_sizes_types
    flag = 3
    clfs, hidden_layer_sizes_types = nnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the min_samples_split with highest accuracy
    activation_types = "NA"
    momentum_values = "NA"
    learning_rates = "NA"
    nn_choice_hiddenlayer_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

    # Now that we have the nn, time for tuning hyperparameters
    # Make a new classifier for this
    clf = MLPClassifier()
    clf.fit(xtrain_val, ytrain_val)
    best_params = nnlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
    print("Best params are: ", best_params)

    # Now do one more fit based on best params above
    final_classifier = MLPClassifier(activation=best_params['activation'],hidden_layer_sizes=best_params['hidden_layer_sizes'], learning_rate=best_params['learning_rate'],momentum=best_params['momentum'],solver='sgd')
    final_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Learning Curves (Neural Nets)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = final_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/nn_learningcurve.png')

    # Now time for final accuracy score for test set
    nnlearner.final_test(final_classifier,xtest,ytest)

    print(datetime.now()-start)

def test_SVM(X_whole, y_whole, X, y):
    

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

    start=datetime.now()

    ### SVMLearner Implementation ###
    svmlearner = svml.SVMLearner(n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

    ########## Initial Learning Curves for Different Kernels ##########

    # Kernel - linear
    # Initial Fit
    initial_classifier = svm.SVC(kernel='linear')
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (SVM - Linear Kernel)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)

    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/svm_learningcurve_initial_linearkernel.png')
    

    # Kernel - poly
    # Initial Fit
    initial_classifier = svm.SVC(kernel='poly')
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (SVM - Poly Kernel)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/svm_learningcurve_initial_polykernel.png')

    # Kernel - rbf
    # Initial Fit
    initial_classifier = svm.SVC(kernel='rbf')
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (SVM - RBF Kernel)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/svm_learningcurve_initial_rbfkernel.png')

    # Kernel - sigmoid
    # Initial Fit
    initial_classifier = svm.SVC(kernel='sigmoid')
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (SVM - Sigmoid Kernel)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/svm_learningcurve_initial_sigmoidkernel.png')


    # Get a list of possible neural nets and their respective kernel types
    flag = 0
    clfs, kernel_types = svmlearner.train(xtrain_val,ytrain_val,flag)
    # Get the SVM that is correlated to the kernel type with highest accuracy
    degree_values = "NA"
    gamma_values = "NA"
    kernel_flag = "NA"
    svm_choice_kernel_based = svmlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,kernel_types,degree_values,gamma_values,flag,kernel_flag)

    # Get a list of possible SVMs and their respective degree values
    flag = 1
    clfs, degree_values = svmlearner.train(xtrain_val,ytrain_val,flag)
    # Get the SVM that is correlated to the degree value with highest accuracy (only for poly kernel)
    kernel_types = "NA"
    gamma_values = "NA"
    kernel_flag = "NA"
    svm_choice_degree_based = svmlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,kernel_types,degree_values,gamma_values,flag,kernel_flag)

    # Get a list of possible SVMs and their respective gamma values
    flag = 2
    clfs_rbf, clfs_poly, clfs_sigmoid, gamma_values = svmlearner.train(xtrain_val,ytrain_val,flag)
    # Get the SVM that is correlated to the gamma with highest accuracy
    kernel_types = "NA"
    degree_values = "NA"
    kernel_flag = 0
    svm_choice_lr_based = svmlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs_rbf,kernel_types,degree_values,gamma_values,flag,kernel_flag)
    kernel_flag = 1
    svm_choice_lr_based = svmlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs_poly,kernel_types,degree_values,gamma_values,flag,kernel_flag)
    kernel_flag = 2
    svm_choice_lr_based = svmlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs_sigmoid,kernel_types,degree_values,gamma_values,flag,kernel_flag)



    # Now that we have the decision tree, time for tuning hyperparameters
    # Make a new classifier for this
    clf = svm.SVC()
    clf.fit(xtrain_val, ytrain_val)
    best_params = svmlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
    print("Best params are: ", best_params)

    # Now do one more fit based on best params above
    final_classifier = svm.SVC(kernel=best_params['kernel'], degree=best_params['degree'], gamma=best_params['gamma'])
    final_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Learning Curves (SVM)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = final_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/svm_learningcurve.png')

    # Now time for final accuracy score for test set
    svmlearner.final_test(final_classifier,xtest,ytest)

    print(datetime.now()-start)

def test_KNN(X_whole, y_whole, X, y):
    

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

    start=datetime.now()

    ### NNLearner Implementation ###
    knnlearner = knn.KNNLearner(n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

    ########## Initial Learning Curves for Different Neighbor Sizes ##########

    # 2 neighbors
    # Initial Fit
    initial_classifier = KNeighborsClassifier(n_neighbors=2)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (KNN - 2 neighbors)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve_initial_2neigh.png')

    # 4 neighbors
    # Initial Fit
    initial_classifier = KNeighborsClassifier(n_neighbors=4)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (KNN - 4 neighbors)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve_initial_4neigh.png')

    # 6 neighbors
    # Initial Fit
    initial_classifier = KNeighborsClassifier(n_neighbors=6)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (KNN - 6 neighbors)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve_initial_6neigh.png')

    # 8 neighbors
    # Initial Fit
    initial_classifier = KNeighborsClassifier(n_neighbors=8)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (KNN - 8 neighbors)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve_initial_8neigh.png')

    # 10 neighbors
    # Initial Fit
    initial_classifier = KNeighborsClassifier(n_neighbors=10)
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (KNN - 10 neighbors)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve_initial_10neigh.png')

    # Get a list of possible knn's and their respective neighbor_types
    flag = 0
    clfs, neighbor_types = knnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the knn that is correlated to the neighbor_type with highest accuracy
    weight_values = "NA"
    algorithm_types = "NA"
    metric_types = "NA"
    p_values = "NA"
    knn_choice_neighbor_based = knnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag)

    # Get a list of possible knns and their respective weight values
    flag = 1
    clfs, weight_values = knnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the knn that is correlated to the weight with highest accuracy
    neighbor_types = "NA"
    algorithm_types = "NA"
    metric_types = "NA"
    p_values = "NA"
    knn_choice_weight_based = knnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag)

    # Get a list of possible knns and their respective algorithm_types
    flag = 2
    clfs, algorithm_types = knnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the knn that is correlated to the algorithm with highest accuracy
    neighbor_types = "NA"
    weight_values = "NA"
    metric_types = "NA"
    p_values = "NA"
    knn_choice_algorithm_based = knnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag)

    # Get a list of possible knns and their respective metric types
    flag = 3
    clfs, metric_types = knnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the knn that is correlated to the metric with highest accuracy
    neighbor_types = "NA"
    weight_values = "NA"
    algorithm_types = "NA"
    p_values = "NA"
    knn_choice_metric_based = knnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag)

    # Get a list of possible knns and their respective p values
    flag = 4
    clfs, p_values = knnlearner.train(xtrain_val,ytrain_val,flag)
    # Get the knn that is correlated to the p value with highest accuracy
    neighbor_types = "NA"
    weight_values = "NA"
    algorithm_types = "NA"
    metric_types = ['minkowski']
    knn_choice_metric_based = knnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag)

    # Now that we have the knn, time for tuning hyperparameters
    # Make a new classifier for this
    clf = KNeighborsClassifier()
    clf.fit(xtrain_val, ytrain_val)
    best_params = knnlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
    print("Best params are: ", best_params)

    # Now do one more fit based on best params above
    final_classifier = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],weights=best_params['weights'], algorithm=best_params['algorithm'],metric=best_params['metric'],p=best_params['p'])
    final_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Learning Curves (KNN)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = final_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/knn_learningcurve.png')

    # Now time for final accuracy score for test set
    knnlearner.final_test(final_classifier,xtest,ytest)

    print(datetime.now()-start)


def test_Boosting(X_whole, y_whole, X, y):
    

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

    start=datetime.now()

    ### Boosting Implementation ###
    boostlearner = boost.BoostingLearner(n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

    ########## Initial Learning Curves for Different Pruning Types ##########

    # ccp_alpha = 0.0
    # Initial Fit
    initial_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0))
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Adaboost - ccp_alpha=0.0)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve_initial_ccpa_0.png')

    # ccp_alpha = 0.0002
    # Initial Fit
    initial_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0002))
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Adaboost - ccp_alpha=0.0002)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve_initial_ccpa_0002.png')

    # ccp_alpha = 0.0004
    # Initial Fit
    initial_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0004))
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Adaboost - ccp_alpha=0.0004)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve_initial_ccpa_0004.png')

    # ccp_alpha = 0.0008
    # Initial Fit
    initial_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0008))
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Adaboost - ccp_alpha=0.0008)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve_initial_ccpa_0008.png')

    # ccp_alpha = 0.0010
    # Initial Fit
    initial_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0010))
    initial_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Initial Learning Curves (Adaboost - ccp_alpha=0.0010)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = initial_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve_initial_ccpa_0010.png')

    # Get a list of possible boostings and their respective alphas
    flag = 0
    clfs, pruning_types = boostlearner.train(xtrain_val,ytrain_val,flag)
    # Get the boosting that is correlated to the alpha with highest accuracy
    number_estimators = "NA"
    learning_rates = "NA"
    boosting_choice_alpha_based = boostlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,pruning_types, number_estimators, learning_rates, flag)

    # Get a list of possible boostings and their respective estimators
    flag = 1
    clfs, number_estimators = boostlearner.train(xtrain_val,ytrain_val,flag)
    # Get the boosting that is correlated to the number of estimators with highest accuracy
    pruning_types = "NA"
    learning_rates = "NA"
    boosting_choice_estimators_based = boostlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,pruning_types, number_estimators, learning_rates, flag)

    # Get a list of possible boostings and their respective learning_rates
    flag = 2
    clfs, learning_rates = boostlearner.train(xtrain_val,ytrain_val,flag)
    # Get the boosting that is correlated to the learning rate with highest accuracy
    pruning_types = "NA"
    number_estimators = "NA"
    boosting_choice_lr_based = boostlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,pruning_types, number_estimators, learning_rates, flag)


    # Now that we have the boosting, time for tuning hyperparameters
    # Make a new classifier for this
    clf = AdaBoostClassifier()
    clf.fit(xtrain_val, ytrain_val)
    best_params = boostlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
    print("Best params are: ", best_params)

    # Now do one more fit based on best params above
    final_classifier = AdaBoostClassifier(base_estimator=best_params['base_estimator'],n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])
    final_classifier.fit(xtrain_val, ytrain_val)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    title = "Learning Curves (Boosting)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = final_classifier
    lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


    lc.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/boosting_learningcurve.png')

    # Now time for final accuracy score for test set
    boostlearner.final_test(final_classifier,xtest,ytest)

    print(datetime.now()-start)


if __name__ == "__main__":  		 
    X_whole, y_whole, X, y = get_data() 	   		     		  		  		    	 		 		   		 		  
    test_DT(X_whole, y_whole, X, y)  
    test_NN(X_whole, y_whole, X, y)	
    test_SVM(X_whole, y_whole, X, y)
    test_KNN(X_whole, y_whole, X, y)
    test_Boosting(X_whole, y_whole, X, y)


    