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
        self.param_dict = {"criterion": ['gini','entropy'], "max_depth": range(1,10), "min_samples_split": range(2,10), "min_samples_leaf": range(1,5)}
        self.grid = 0

    def train(self, X_train, y_train, flag):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training Decision Tree Model...")

        if flag == 0:

            # Use cost_complexity_pruning_path to get effective alphas (these is just what values of ccp_alpha could be appropriate)
            # We can also get corresponding leaf impurities if desired (not needed for now)
            # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
            path = self.clf.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            ccp_alphas = ccp_alphas[0::5]

            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, y_train)
                clfs.append(clf)
            # print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            #     clfs[-1].tree_.node_count, ccp_alphas[-1]))
            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]

            return clfs, ccp_alphas

        if flag == 1:

            possible_depths = range(1,25)

            clfs = []
            for depth in possible_depths:
                clf = DecisionTreeClassifier(random_state=0, max_depth=depth)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, possible_depths

        if flag == 2:

            possible_min_samples_leaf = range(1,20)

            clfs = []
            for min_samples_leaf in possible_min_samples_leaf:
                clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=min_samples_leaf)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, possible_min_samples_leaf

        if flag == 3:

            possible_min_samples_split = range(2,20)

            clfs = []
            for min_samples_split in possible_min_samples_split:
                clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_samples_split)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, possible_min_samples_split

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



    def test(self, X_test,X_train, y_test, y_train, clfs, alphas, depths, min_samples_leafs, min_samples_splits, flag):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing Decision Tree Model...")

        if flag == 0:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Alpha combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best Alpha (Highest Accuracy, Test Validation Set): ", alphas[self.accuracy_score_test.index(max(self.accuracy_score_test))])

            plt.figure()
            plt.plot(alphas, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(alphas, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Alpha')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Alpha Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/alpha_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        if flag == 1:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Depth combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best Depth (Highest Accuracy, Test Validation Set): ", depths[self.accuracy_score_test.index(max(self.accuracy_score_test))])

            plt.figure()
            plt.plot(depths, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(depths, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Depth')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Depth Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/depth_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        if flag == 2:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Depth combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best min_sample_leaf (Highest Accuracy, Test Validation Set): ", min_samples_leafs[self.accuracy_score_test.index(max(self.accuracy_score_test))])

            plt.figure()
            plt.plot(min_samples_leafs, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(min_samples_leafs, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('min_sample_leaf')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs min_sample_leaf Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/minsampleleaf_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        if flag == 3:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Depth combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best min_sample_split (Highest Accuracy, Test Validation Set): ", min_samples_splits[self.accuracy_score_test.index(max(self.accuracy_score_test))])

            plt.figure()
            plt.plot(min_samples_splits, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(min_samples_splits, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('min_sample_split')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs min_sample_split Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/minsamplesplit_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

    def tune_hyperparameters(self, final_dt, xtrain, ytrain):
        self.grid = GridSearchCV(final_dt, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        print(self.grid.best_params_)
