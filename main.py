from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import display
from IPython.display import Image  
import pydotplus
import graphviz
from sklearn import tree
import numpy as np

import DTLearner as dt
import NNLearner as nn
import SVMLearner as svml

from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve

from sklearn.neural_network import MLPClassifier
from sklearn import svm

def get_data():
    # Fetch and split Data here
    mnist_data = fetch_openml('mnist_784')
    #print(mnist_data.keys())
    X_whole, y_whole = mnist_data['data'], mnist_data['target']

    # Take a subset of the data (10%)
    X = X_whole[0::10]
    y = y_whole[0::10]

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


if __name__ == "__main__":  		 
    X_whole, y_whole, X, y = get_data() 	   		     		  		  		    	 		 		   		 		  
    #test_DT(X_whole, y_whole, X, y)  
    #test_NN(X_whole, y_whole, X, y)	
    test_SVM(X_whole, y_whole, X, y)


    