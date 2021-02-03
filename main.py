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

import DTLearner as dt

def test_code():
    


    # Fetch and split Data here
    mnist_data = fetch_openml('mnist_784')
    #print(mnist_data.keys())
    X, y = mnist_data['data'], mnist_data['target']
    X = X[0::10]
    y = y[0::10]
    xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,shuffle = False,random_state =7)

    start=datetime.now()

    dtlearner = dt.DTLearner(leaf_size=1, n_folds=2, verbose=True)  
    clfs, alphas = dtlearner.train(xtrain,ytrain)
    dtlearner.test(xtest,xtrain,ytest,ytrain,clfs,alphas)

    print(datetime.now()-start)


if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  	


    