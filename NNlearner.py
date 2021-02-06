from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import display
from IPython.display import Image  
import pydotplus
import graphviz
from sklearn import tree


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier


class NNLearner(object):

    def __init__(self, n_folds=10, verbose=False):
        self.n_folds = n_folds
        self.clf = MLPClassifier()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        self.param_dict = {"activation": ['identity', 'logistic', 'tanh', 'relu'], "momentum": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "learning_rate": ['constant', 'invscaling', 'adaptive'], "hidden_layer_sizes": (100,)}
        self.grid = 0

        # Write data to file for easy analysis
        self.f = open("nn_info.txt", "a")
        self.f.write("\n")
        self.f.write(str(datetime.datetime.now()))


    def train(self, X_train, y_train, flag):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training Neural Network Model...")
            self.f.write("Training Neural Network Model...")

        if flag == 0:

            clfs = []
            activation_types = ['identity', 'logistic', 'tanh', 'relu']
            for activation_type in activation_types:
                clf = MLPClassifier(activation=activation_type)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, activation_types

        # if flag == 1:

        #     possible_depths = range(1,25)

        #     clfs = []
        #     for depth in possible_depths:
        #         clf = DecisionTreeClassifier(random_state=0, max_depth=depth)
        #         clf.fit(X_train, y_train)
        #         clfs.append(clf)

        #     return clfs, possible_depths

        # if flag == 2:

        #     possible_min_samples_leaf = range(1,20)

        #     clfs = []
        #     for min_samples_leaf in possible_min_samples_leaf:
        #         clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=min_samples_leaf)
        #         clf.fit(X_train, y_train)
        #         clfs.append(clf)

        #     return clfs, possible_min_samples_leaf

        # if flag == 3:

        #     possible_min_samples_split = range(2,20)

        #     clfs = []
        #     for min_samples_split in possible_min_samples_split:
        #         clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_samples_split)
        #         clf.fit(X_train, y_train)
        #         clfs.append(clf)

        #     return clfs, possible_min_samples_split




    def test(self, X_test,X_train, y_test, y_train, clfs, activation_types, hidden_layer_sizes_types, learning_rates, momentum_values, flag):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing Neural Network Model...")

        if flag == 0:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Activation_Type combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best Activation Type (Highest Accuracy, Test Validation Set): ", activation_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Activation Type (Highest Accuracy, Test Validation Set): " + str(activation_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(activation_types, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(activation_types, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Activation Type')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Activation Type')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/activationtype_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        # if flag == 1:
        #     self.accuracy_score_train = []
        #     self.accuracy_score_test = []

        #     for clf in clfs:
        #         predictions_train = clf.predict(X_train)
        #         predictions_test = clf.predict(X_test)

        #         self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
        #         self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


        #     # Print out best Accuracy/Depth combination
        #     print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
        #     print("Best Depth (Highest Accuracy, Test Validation Set): ", depths[self.accuracy_score_test.index(max(self.accuracy_score_test))])
        #     self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
        #     self.f.write("Best Depth (Highest Accuracy, Test Validation Set): " + str(depths[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

        #     plt.figure()
        #     plt.plot(depths, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
        #     plt.plot(depths, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
        #     plt.xlabel('Depth')
        #     plt.ylabel('Accuracy')
        #     plt.title('Accuracy vs Depth Value')
        #     plt.legend()
        #     plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/depth_vs_accuracy.png')

        #     return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        # if flag == 2:
        #     self.accuracy_score_train = []
        #     self.accuracy_score_test = []

        #     for clf in clfs:
        #         predictions_train = clf.predict(X_train)
        #         predictions_test = clf.predict(X_test)

        #         self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
        #         self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


        #     # Print out best Accuracy/Depth combination
        #     print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
        #     print("Best min_sample_leaf (Highest Accuracy, Test Validation Set): ", min_samples_leafs[self.accuracy_score_test.index(max(self.accuracy_score_test))])
        #     self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
        #     self.f.write("Best min_sample_leaf (Highest Accuracy, Test Validation Set): " + str(min_samples_leafs[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

        #     plt.figure()
        #     plt.plot(min_samples_leafs, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
        #     plt.plot(min_samples_leafs, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
        #     plt.xlabel('min_sample_leaf')
        #     plt.ylabel('Accuracy')
        #     plt.title('Accuracy vs min_sample_leaf Value')
        #     plt.legend()
        #     plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/minsampleleaf_vs_accuracy.png')

        #     return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        # if flag == 3:
        #     self.accuracy_score_train = []
        #     self.accuracy_score_test = []

        #     for clf in clfs:
        #         predictions_train = clf.predict(X_train)
        #         predictions_test = clf.predict(X_test)

        #         self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
        #         self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


        #     # Print out best Accuracy/Depth combination
        #     print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
        #     print("Best min_sample_split (Highest Accuracy, Test Validation Set): ", min_samples_splits[self.accuracy_score_test.index(max(self.accuracy_score_test))])
        #     self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
        #     self.f.write("Best min_sample_split (Highest Accuracy, Test Validation Set): " + str(min_samples_splits[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

        #     plt.figure()
        #     plt.plot(min_samples_splits, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
        #     plt.plot(min_samples_splits, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
        #     plt.xlabel('min_sample_split')
        #     plt.ylabel('Accuracy')
        #     plt.title('Accuracy vs min_sample_split Value')
        #     plt.legend()
        #     plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/DT/minsamplesplit_vs_accuracy.png')

        #     return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

    def tune_hyperparameters(self, final_dt, xtrain, ytrain):
        self.grid = GridSearchCV(final_dt, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        self.f.write("Best Params from GridSearchCV: " + str(self.grid.best_params_))
        self.f.close()
        return self.grid.best_params_
