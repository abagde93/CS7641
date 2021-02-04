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

def test_code():
    

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

    # Split the initial data
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,shuffle = False,random_state =7)

    start=datetime.now()

    ### DTLearner Implementation ###
    dtlearner = dt.DTLearner(leaf_size=1, n_folds=3, verbose=True)  

    # Create a validation set - do another train/test split on the training data
    xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,shuffle = False,random_state =7)

    # Get a list of possible decision trees and their respective alphas
    flag = 0
    clfs, alphas = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the alpha with highest accuracy
    depths = "NA"
    dt_choice = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,flag)

    # Get a list of possible decision trees and their respective depths
    flag = 1
    clfs, depths = dtlearner.train(xtrain_val,ytrain_val,flag)
    # Get the decision tree that is correlated to the depth with highest accuracy
    alphas = "NA"
    dt_choice = dtlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,alphas,depths,flag)

    # # Now that we have the decision tree, time for tuning hyperparameters
    # dtlearner.tune_hyperparameters(dt_choice, xtrain, ytrain)


    print(datetime.now()-start)


if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  	


    