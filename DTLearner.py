from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import GridSearchCV, train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import display
from IPython.display import Image  
import pydotplus
import graphviz
from sklearn import tree


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


class DTLearner(object):

    def __init__(self, leaf_size=1, n_folds=10, verbose=False):
        self.leaf_size = leaf_size
        self.n_folds = n_folds
        self.cv_scores = []
        self.clf = DecisionTreeClassifier()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        # NOTE: Add alpha to param_dict, figure out wtf it is
        self.param_dict = {"criterion": ['gini','entropy'], "max_depth": range(1,8), "min_samples_split": range(2,8), "min_samples_leaf": range(1,7)}
        self.grid = 0

    def train(self, X_train, y_train):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training Decision Tree Model...")

        path = self.clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        ccp_alphas = ccp_alphas[0::200]
        impurities = impurities[0::200]
        print(ccp_alphas)
        self.param_dict['ccp_alpha'] = ccp_alphas

        self.grid = GridSearchCV(self.clf, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(X_train, y_train)

        print(self.grid.best_params_)

        # fig, ax = plt.subplots()
        # ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        # ax.set_xlabel("effective alpha")
        # ax.set_ylabel("total impurity of leaves")
        # ax.set_title("Total Impurity vs effective alpha for training set")
        # plt.show()


    def test(self, X_test, y_test):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing Decision Tree Model...")

        self.predictions = self.grid.predict(X_test)
        self.accuracy_score_test = accuracy_score(y_test, self.predictions)
        self.accuracy_score_train = accuracy_score(y_train, self.predictions)
        print("Train Accuracy:", self.accuracy_score_test)
        print("Test Accuracy:", self.accuracy_score)


