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

from sklearn.neighbors import KNeighborsClassifier


class KNNLearner(object):

    def __init__(self, n_folds=10, verbose=False):
        self.n_folds = n_folds
        self.clf = KNeighborsClassifier()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        self.param_dict = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10], "weights": ['uniform','distance'], "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'], "metric": ['minkowski','euclidean','manhattan'], "p": [2,3,4]}
        self.grid = 0

        # Write data to file for easy analysis
        self.f = open("knn_info.txt", "a")
        self.f.write("\n")
        self.f.write(str(datetime.now()))


    def train(self, X_train, y_train, flag):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training KNN Model...")
            self.f.write("Training KNN Model...")

        if flag == 0:

            clfs = []
            neighbor_types = ['identity', 'logistic', 'tanh', 'relu']
            for neighbor in neighbor_types:
                clf = KNeighborsClassifier(n_neighbors=neighbor)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, neighbor_types

        if flag == 1:

            weight_values = ['uniform','distance']

            clfs = []
            for weight in weight_values:
                clf = KNeighborsClassifier(weights=weight)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, weight_values

        if flag == 2:

            algorithm_types = ['auto', 'ball_tree', 'kd_tree', 'brute']

            clfs = []
            for algorithm in algorithm_types:
                clf = KNeighborsClassifier(algorithm=algorithm)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, algorithm_types

        if flag == 3:

            metric_types = ['minkowski','euclidean','manhattan']

            clfs = []
            for metric in metric_types:
                clf = KNeighborsClassifier(metric=metric)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, metric_types

        if flag == 4:

            p_values = [2,3,4]

            clfs = []
            for p_value in p_values:
                clf = KNeighborsClassifier(p=p_value,metric='minkowski')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, p_values




    def test(self, X_test,X_train, y_test, y_train, clfs, neighbor_types, weight_values, algorithm_types, metric_types, p_values, flag):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing KNN Model...")

        if flag == 0:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/neighbors combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best Neighbor Size (Highest Accuracy, Test Validation Set): ", neighbor_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Neighbor Size (Highest Accuracy, Test Validation Set): " + str(neighbor_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(neighbor_types, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(neighbor_types, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Neighbor Size')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Neighbor Size')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/neighborsize_vs_accuracy.png')

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
            print("Best Weight Value (Highest Accuracy, Test Validation Set): ", weight_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Weight Value (Highest Accuracy, Test Validation Set): " + str(weight_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(momentum_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(momentum_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Weight Value')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Weight Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/weight_vs_accuracy.png')

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
            print("Best Algorithm (Highest Accuracy, Test Validation Set): ", algorithm_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Algorithm (Highest Accuracy, Test Validation Set): " + str(algorithm_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(algorithm_types, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(algorithm_types, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Algorithm')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Algorithm')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/algorithm_vs_accuracy.png')

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
            print("Best Metric (Highest Accuracy, Test Validation Set): ", metric_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Metric (Highest Accuracy, Test Validation Set): " + str(metric_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(metric_types, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(metric_types, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Metric')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Metric')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/metrictypes_vs_accuracy.png')

        if flag == 4:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/Depth combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best P Value (minkowski metric) (Highest Accuracy, Test Validation Set): ", p_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best P Value (minkowski metric) (Highest Accuracy, Test Validation Set): " + str(p_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(p_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(p_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('P Value')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs P Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/KNN/pvalue_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

    def tune_hyperparameters(self, final_knn, xtrain, ytrain):
        self.grid = GridSearchCV(final_nn, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        self.f.write("Best Params from GridSearchCV: " + str(self.grid.best_params_))
        return self.grid.best_params_

    def final_test(self, clf, xtest, ytest):
        prediction_test = clf.predict(xtest)
        print(accuracy_score(ytest, prediction_test))
        self.f.write("Final Accuracy Score (Test Set): " + str(accuracy_score(ytest, prediction_test)))
        self.f.close()
