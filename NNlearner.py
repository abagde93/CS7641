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


class NNLearner(object):

    def __init__(self, n_folds=10, verbose=False):
        self.n_folds = n_folds
        self.clf = MLPClassifier()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        self.param_dict = {"activation": ['identity', 'logistic', 'tanh', 'relu'], "momentum": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "learning_rate": ['constant', 'invscaling', 'adaptive'], "hidden_layer_sizes": [(50,),(100,),(200,),(50, 50),(100, 100), (200, 200), (50,50,50), (100,100,100), (200,200,200)], "solver": ['sgd']}
        self.grid = 0

        # Write data to file for easy analysis
        self.f = open("nn_info.txt", "a")
        self.f.write("\n")
        self.f.write(str(datetime.now()))


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
                clf = MLPClassifier(activation=activation_type,solver='sgd')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, activation_types

        if flag == 1:

            momentum_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            clfs = []
            for value in momentum_values:
                clf = MLPClassifier(momentum=value,solver='sgd')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, momentum_values

        if flag == 2:

            learning_rates = ['constant', 'invscaling', 'adaptive']

            clfs = []
            for learning_rate in learning_rates:
                clf = MLPClassifier(learning_rate=learning_rate,solver='sgd')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, learning_rates

        if flag == 3:

            hidden_layer_sizes_types = [(50,),(100,),(200,),(50, 50),(100, 100), (200, 200), (50,50,50), (100,100,100), (200,200,200)]
            #hidden_layer_sizes_types_for_plot = ["(10,)","(100,)","(10,50,10)","(20,20,20)"]

            clfs = []
            for hidden_layer_size in hidden_layer_sizes_types:
                clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size,solver='sgd')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, hidden_layer_sizes_types




    def test(self, X_test,X_train, y_test, y_train, clfs, activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag):
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
            print("Best Momentum Value (Highest Accuracy, Test Validation Set): ", momentum_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Momentum Value (Highest Accuracy, Test Validation Set): " + str(momentum_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(momentum_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(momentum_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Momentum Value')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Momentum Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/momentum_vs_accuracy.png')

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
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/learningrate_vs_accuracy.png')

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
            print("Best Hidden Layer Sizes (Highest Accuracy, Test Validation Set): ", hidden_layer_sizes_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Hidden Layer Sizes (Highest Accuracy, Test Validation Set): " + str(hidden_layer_sizes_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure(figsize=(11,6))
            hl_plot_array = []
            for item in hidden_layer_sizes_types:
                hl_plot_array.append(str(item))
            plt.plot(hl_plot_array, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(hl_plot_array, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('hidden_layer_sizes_types')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs hidden_layer_sizes_types')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/NN/hiddenlayersizestypes_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

    def tune_hyperparameters(self, final_nn, xtrain, ytrain):
        self.grid = GridSearchCV(final_nn, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        self.f.write("Best Params from GridSearchCV: " + str(self.grid.best_params_))
        return self.grid.best_params_

    def final_test(self, clf, xtest, ytest):
        prediction_test = clf.predict(xtest)
        print(accuracy_score(ytest, prediction_test))
        self.f.write("Final Accuracy Score (Test Set): " + str(accuracy_score(ytest, prediction_test)))
        self.f.close()
