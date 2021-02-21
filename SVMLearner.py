from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn import svm


class SVMLearner(object):

    def __init__(self, n_folds=10, verbose=False):
        self.n_folds = n_folds
        self.clf = svm.SVC()
        self.predictions = []
        self.accuracy_score = 0.0
        self.verbose = verbose

        self.param_dict = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "degree": [1,2,3,4,5], "gamma": ['scale', 'auto']}
        self.grid = 0

        # Write data to file for easy analysis
        self.f = open("svm_info.txt", "a")
        self.f.write("\n")
        self.f.write(str(datetime.now()))


    def train(self, X_train, y_train, flag):
        '''

        :param X_train: training data
        :param y_train: training labels
        :return:
        '''

        if self.verbose:
            print("Training SVM Model...")
            self.f.write("Training SVM Model...")

        if flag == 0:

            clfs = []
            kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
            for kernel_type in kernel_types:
                clf = svm.SVC(kernel=kernel_type)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, kernel_types

        if flag == 1:

            degree_values = [1,2,3,4,5]

            clfs = []
            for value in degree_values:
                clf = svm.SVC(degree=value,kernel='poly')
                clf.fit(X_train, y_train)
                clfs.append(clf)

            return clfs, degree_values

        if flag == 2:

            gamma_values = ['scale', 'auto']

            clfs_poly = []
            clfs_rbf= []
            clfs_sigmoid = []
            for gamma_value in gamma_values:
                clf = svm.SVC(gamma=gamma_value,kernel='rbf')
                clf.fit(X_train, y_train)
                clfs_rbf.append(clf)
            for gamma_value in gamma_values:
                clf = svm.SVC(gamma=gamma_value,kernel='poly')
                clf.fit(X_train, y_train)
                clfs_poly.append(clf)
            for gamma_value in gamma_values:
                clf = svm.SVC(gamma=gamma_value,kernel='sigmoid')
                clf.fit(X_train, y_train)
                clfs_sigmoid.append(clf)

            return clfs_rbf, clfs_poly, clfs_sigmoid, gamma_values





    def test(self, X_test,X_train, y_test, y_train, clfs, kernel_types, degree_values, gamma_values, flag, kernel_flag):
        '''

        :param X_test: test data
        :param y_test: test labels
        :return:
        '''

        if self.verbose:
            print("Testing SVM Model...")

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
            print("Best Kernel (Highest Accuracy, Test Validation Set): ", kernel_types[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Kernel (Highest Accuracy, Test Validation Set): " + str(kernel_types[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(kernel_types, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(kernel_types, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Kernel')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Kernel Type')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/kernel_vs_accuracy.png')

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
            print("Best Degree Value (Highest Accuracy, Test Validation Set): ", degree_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
            self.f.write("Best Accuracy Score (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
            self.f.write("Best Degree Value (Highest Accuracy, Test Validation Set): " + str(degree_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

            plt.figure()
            plt.plot(degree_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
            plt.plot(degree_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
            plt.xlabel('Degree Value')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Degree Value')
            plt.legend()
            plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/degree_vs_accuracy.png')

            return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]

        if flag == 2:
            self.accuracy_score_train = []
            self.accuracy_score_test = []

            for clf in clfs:
                predictions_train = clf.predict(X_train)
                predictions_test = clf.predict(X_test)

                self.accuracy_score_train.append(accuracy_score(y_train, predictions_train))
                self.accuracy_score_test.append(accuracy_score(y_test, predictions_test))


            if kernel_flag == 0:

                print("Best Accuracy Score (rbf kernel) (Test Validation Set): ", max(self.accuracy_score_test))
                print("Best Gamma Value (rbf kernel) (Highest Accuracy, Test Validation Set): ", gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
                self.f.write("Best Accuracy Score (rbf kernel) (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
                self.f.write("Best Gamma Value (rbf kernel) (Highest Accuracy, Test Validation Set): " + str(gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

                plt.figure()
                plt.plot(gamma_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
                plt.plot(gamma_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
                plt.xlabel('Gamma Value')
                plt.ylabel('Accuracy')
                plt.title('Accuracy vs Gamma Value (rbf kernel)')
                plt.legend()
                plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/gamma_vs_accuracy_rbf.png')

                return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]
            
            if kernel_flag == 1:

                print("Best Accuracy Score (poly kernel) (Test Validation Set): ", max(self.accuracy_score_test))
                print("Best Gamma Value (poly kernel) (Highest Accuracy, Test Validation Set): ", gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
                self.f.write("Best Accuracy Score (poly kernel) (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
                self.f.write("Best Gamma Value (poly kernel) (Highest Accuracy, Test Validation Set): " + str(gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

                plt.figure()
                plt.plot(gamma_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
                plt.plot(gamma_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
                plt.xlabel('Gamma Value')
                plt.ylabel('Accuracy')
                plt.title('Accuracy vs Gamma Value (poly kernel)')
                plt.legend()
                plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/gamma_vs_accuracy_poly.png')

                return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]
            
            if kernel_flag == 2:

                print("Best Accuracy Score (sigmoid kernel) (Test Validation Set): ", max(self.accuracy_score_test))
                print("Best Gamma Value (sigmoid kernel) (Highest Accuracy, Test Validation Set): ", gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))])
                self.f.write("Best Accuracy Score (sigmoid kernel) (Test Validation Set): " + str(max(self.accuracy_score_test)) + "\n")
                self.f.write("Best Gamma Value (sigmoid kernel) (Highest Accuracy, Test Validation Set): " + str(gamma_values[self.accuracy_score_test.index(max(self.accuracy_score_test))]) + "\n")

                plt.figure()
                plt.plot(gamma_values, self.accuracy_score_train, label = 'Accuracy Score (Training Validation Set)')
                plt.plot(gamma_values, self.accuracy_score_test, label = 'Accuracy Score (Test Validation Set)')
                plt.xlabel('Gamma Value')
                plt.ylabel('Accuracy')
                plt.title('Accuracy vs Gamma Value (sigmoid kernel)')
                plt.legend()
                plt.savefig('/Users/ajinkya.bagde/Desktop/AS1_Figs/SVM/gamma_vs_accuracy_sigmoid.png')

                return clfs[self.accuracy_score_test.index(max(self.accuracy_score_test))]



    def tune_hyperparameters(self, final_svm, xtrain, ytrain):
        self.grid = GridSearchCV(final_svm, param_grid = self.param_dict, cv=self.n_folds, verbose=1, n_jobs=-1)
        self.grid.fit(xtrain, ytrain)

        self.f.write("Best Params from GridSearchCV: " + str(self.grid.best_params_))
        return self.grid.best_params_

    def final_test(self, clf, xtest, ytest):
        prediction_test = clf.predict(xtest)
        print(accuracy_score(ytest, prediction_test))
        self.f.write("Final Accuracy Score (Test Set): " + str(accuracy_score(ytest, prediction_test)))
        self.f.close()
