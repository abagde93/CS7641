from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

class BoostingLearner(object):

    def __init__(self, n_folds=10, verbose=False):
        self.n_folds = n_folds
        self.clf = AdaBoostClassifier()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        self.param_dict = {"base_estimator": [DecisionTreeClassifier(ccp_alpha=0.0),DecisionTreeClassifier(ccp_alpha=0.0002),DecisionTreeClassifier(ccp_alpha=0.0004),DecisionTreeClassifier(ccp_alpha=0.0006),DecisionTreeClassifier(ccp_alpha=0.0008),DecisionTreeClassifier(ccp_alpha=0.001)], "n_estimators": [20,40,60,80,100,120,140,160,180,200], "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        self.grid = 0

        # Write data to file for easy analysis
        self.f = open("boosting_info.txt", "a")
        self.f.write("\n")
        self.f.write(str(datetime.now()))


    def train(self, X_train, y_train, flag):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training Boosting Model...")
            self.f.write("Training Boosting Model...")

        if flag == 0:

            clfs = []
            pruning_types = [DecisionTreeClassifier(ccp_alpha=0.0),DecisionTreeClassifier(ccp_alpha=0.0002),DecisionTreeClassifier(ccp_alpha=0.0004),DecisionTreeClassifier(ccp_alpha=0.0006),DecisionTreeClassifier(ccp_alpha=0.0008),DecisionTreeClassifier(ccp_alpha=0.001)]
            for tree in pruning_types:
                clf = AdaBoostClassifier(base_estimator=tree)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, pruning_types

        if flag == 1:

            number_estimators = [20,40,60,80,100,120,140,160,180,200]

            clfs = []
            for estimator in number_estimators:
                clf = AdaBoostClassifier(n_estimators=estimator)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, number_estimators

        if flag == 2:

            learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            clfs = []
            for learning_rate in learning_rates:
                clf = AdaBoostClassifier(learning_rate=learning_rate)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, learning_rates




    def test(self, X_test,X_train, y_test, y_train, clfs, pruning_types, number_estimators, learning_rates, flag):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing Boosting Model...")

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
            print("Best Alpha (Highest Accuracy, Test Validation Set): ", pruning_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Alpha (Highest Accuracy, Test Validation Set): " + str(pruning_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            pruning_alphas = [0.0,0.0002,0.0004,0.0006,0.0008,0.0010]

            plt.figure()
            plt.plot(pruning_alphas, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(pruning_alphas, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Alpha Value')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Alpha')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/alpha_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        if flag == 1:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            # Print out best Accuracy/estimators combination
            print("Best Accuracy Score (Test Validation Set): ", max(self.accuracy_score_test))
            print("Best Estimators Value (Highest Accuracy, Test Validation Set): ", number_estimators[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Estimators Value (Highest Accuracy, Test Validation Set): " + str(number_estimators[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(number_estimators, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(number_estimators, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Weight Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/estimators_vs_accuracy.png')

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
            print("Best Learning Rate (Highest Accuracy, Test Validation Set): ", learning_rates[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Learning Rate (Highest Accuracy, Test Validation Set): " + str(learning_rates[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(learning_rates, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(learning_rates, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Learning Rate')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Learning Rate')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/Boosting/learningrate_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]



    def tune_hyperparameters(self, final_boosting, xtrain, ytrain):
        self.grid = GridSearchCV(final_boosting, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        self.f.write("Best Params from GridSearchCV: " + str(self.grid.best_params_))
        return self.grid.best_params_

    def final_test(self, clf, xtest, ytest):
        prediction_test = clf.predict(xtest)
        print(accuracy_score(ytest, prediction_test))
        self.f.write("Final Accuracy Score (Test Set): " + str(accuracy_score(ytest, prediction_test)))
        self.f.close()
