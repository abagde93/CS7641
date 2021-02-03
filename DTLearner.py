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
        self.param_dict = {"criterion": ['gini','entropy'], "max_depth": range(1,3), "min_samples_split": range(2,4), "min_samples_leaf": range(1,4)}
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
        ccp_alphas = ccp_alphas[0::20]
        impurities = impurities[0::20]
        #print(ccp_alphas)
        #self.param_dict['ccp_alpha'] = ccp_alphas

        # self.grid = GridSearchCV(self.clf, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        # self.grid.fit(X_train, y_train)

        # print(self.grid.best_params_)

        # Total impurity of leaves vs effective alphas of pruned tree
        # fig, ax = plt.subplots()
        # ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        # ax.set_xlabel("effective alpha")
        # ax.set_ylabel("total impurity of leaves")
        # ax.set_title("Total Impurity vs effective alpha for training set")

        # Number of Nodes vs alpha and Depth vs Alpha
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
        print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]))

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        # node_counts = [clf.tree_.node_count for clf in clfs]
        # depth = [clf.tree_.max_depth for clf in clfs]
        # fig, ax = plt.subplots(2,1)
        # ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        # ax[0].set_xlabel("alpha")
        # ax[0].set_ylabel("number of nodes")
        # ax[0].set_title("Number of nodes vs alpha")
        # ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        # ax[1].set_xlabel("alpha")
        # ax[1].set_ylabel("depth of tree")
        # ax[1].set_title("Depth vs alpha")
        # fig.tight_layout()
        # plt.show()

        return clfs, ccp_alphas




    def test(self, X_test,X_train, y_test, y_train, clfs, alphas):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing Decision Tree Model...")

        self.accuracy_score_train = []
        self.accuracy_score_test = []

        for clf in clfs:
            predictions_train = clf.predict(X_train)
            predictions_test = clf.predict(X_test)

            self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
            self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))

        for i in range(0, len(alphas)):
            print("Alpha Value: ", alphas[i])
            print("Train Accuracy:", self.accuracy_score_train[i])
            print("Test Accuracy:", self.accuracy_score_test[i])

        plt.plot(alphas, self.accuracy_score_train, label = 'Accuracy Score (Training Set)')
        plt.plot(alphas, self.accuracy_score_test, label = 'Accuracy Score (Test Set)')
        plt.xlabel('Alpha')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Alpha Value')
        plt.legend()
        plt.show()


