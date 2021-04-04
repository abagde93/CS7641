from datetime import datetime
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import numpy as np



from sklearn.model_selection import ShuffleSplit

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans



import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import accuracy_score
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import mean_squared_error

import pandas as pd
import plotly.express as px

from sklearn.random_projection import GaussianRandomProjection

from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition import TruncatedSVD

import math

import NNLearner as nn
from plot_learning_curve import plot_learning_curve


def import_data(dataset_name):
    '''
    Import a named dataset using scikit-learn's built in fetech_openml function
    :return: the specified dataset
    '''
    dataset = fetch_openml(dataset_name, data_home="data/")
    return dataset

def kmeans_analysis(x_train, y_train, plot_path):

    # Numbers of clusters to test 
    clusters = list(range(2,70,1))

    # Plotting Inertia
    # What is Inertia? - The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares
    # Inertia can be recognized as a measure of how internally coherent clusters are.
    # This is the elbow method of determining clusters
    # https://predictivehacks.com/k-means-elbow-method-code-for-python/
    inertia = {}
    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, max_iter=1000, random_state=4469).fit(x_train)
        inertia[cluster] = kmeans.inertia_
    plt.figure()
    plt.plot(list(inertia.keys()), list(inertia.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig(plot_path + '/clusters_vs_inertia')

    # Its hard to see an elbow in this plot, so lets look at other metrics
    # Specifically Silhoutte Coefficient, Homogeneity, and Completeness 

    silhoutte_coefficients_scores = {}
    homogeneity_scores = {}
    completeness_scores = {}
    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, max_iter=1000, random_state=4469).fit(x_train)
        label = kmeans.labels_
        silhoutte_coefficient = silhouette_score(x_train, label, metric='euclidean')
        silhoutte_coefficients_scores[cluster] = silhoutte_coefficient
        homo_score = homogeneity_score(y_train, label)
        homogeneity_scores[cluster] = homo_score
        comp_score = completeness_score(y_train, label)
        completeness_scores[cluster] = comp_score
    plt.figure()
    plt.plot(list(silhoutte_coefficients_scores.keys()), list(silhoutte_coefficients_scores.values()), label='silhoutte_coefficients_score')
    plt.plot(list(homogeneity_scores.keys()), list(homogeneity_scores.values()), label='homogeneity_score')
    plt.plot(list(completeness_scores.keys()), list(completeness_scores.values()), label='completeness_score')
    plt.xlabel("Number of clusters")
    plt.title("Clustering Performance Evaluation")
    plt.legend()
    plt.savefig(plot_path + '/clustering_performance_evaluation_kmeans')

    # From this we picked a cluster size of 45 as optimal for Spambase
    # Lets see the percentage of clusters that align with classes
    # 14.3% for Kmeans (not great)
    # TODO: PLay around with this and optimal cluster size
    kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=4469).fit(x_train)
    print("Accuracy score is: ", accuracy_score(label, y_train))


def EM_analysis(x_train, y_train, plot_path):

    # Numbers of clusters to test 
    clusters = list(range(2,70,1))

    # Its hard to see an elbow in this plot, so lets look at other metrics
    # Specifically Silhoutte Coefficient, Homogeneity, and Completeness 

    silhoutte_coefficients_scores = {}
    homogeneity_scores = {}
    completeness_scores = {}
    for cluster in clusters:
        gmm = GaussianMixture(n_components=cluster, max_iter=1000, random_state=4469).fit(x_train)
        label = gmm.predict(x_train)
        silhoutte_coefficient = silhouette_score(x_train, label, metric='euclidean')
        silhoutte_coefficients_scores[cluster] = silhoutte_coefficient
        homo_score = homogeneity_score(y_train, label)
        homogeneity_scores[cluster] = homo_score
        comp_score = completeness_score(y_train, label)
        completeness_scores[cluster] = comp_score
    plt.figure()
    plt.plot(list(silhoutte_coefficients_scores.keys()), list(silhoutte_coefficients_scores.values()), label='silhoutte_coefficients_score')
    plt.plot(list(homogeneity_scores.keys()), list(homogeneity_scores.values()), label='homogeneity_score')
    plt.plot(list(completeness_scores.keys()), list(completeness_scores.values()), label='completeness_score')
    plt.xlabel("Number of clusters")
    plt.title("Clustering Performance Evaluation")
    plt.legend()
    plt.savefig(plot_path + '/clustering_performance_evaluation_EM')

    # From this we picked a cluster size of 45 as optimal for Spambase
    # Lets see the percentage of clusters that align with classes
    # 14.3% for Kmeans (not great)
    # TODO: PLay around with this and optimal cluster size
    gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(x_train)
    print("Accuracy score is: ", accuracy_score(label, y_train))


 
def pca(X_train, plot_path):
    plt.figure()
    pca = PCA().fit(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Variance')
    plt.savefig(plot_path + '/pca_variance')
    plt.close()

def pca_analysis(X, y, plot_path):

    feat_cols = list(X)
    df = X#pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    # For reproducability of the results
    #np.random.seed(42)
    #rndperm = np.random.permutation(df.shape[0])
    label_list = df['y'].tolist()
    label_list = list(map(int, label_list))
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"], 
        ys=df["pca-two"], 
        zs=df["pca-three"], 
        c=label_list,
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.savefig(plot_path + '/pca_3D')

def ica_analysis(X, y, plot_path):

    feat_cols = list(X)
    df = X#pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    # For reproducability of the results
    #np.random.seed(42)
    #rndperm = np.random.permutation(df.shape[0])
    label_list = df['y'].tolist()
    label_list = list(map(int, label_list))
    print(label_list)
    ica = ICA(n_components=3)
    ica_result = ica.fit_transform(df[feat_cols].values)
    df['ica-one'] = ica_result[:,0]
    df['ica-two'] = ica_result[:,1] 
    df['ica-three'] = ica_result[:,2]
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df["ica-one"], 
        ys=df["ica-two"], 
        zs=df["ica-three"], 
        c=label_list,
        cmap='tab10'
    )
    ax.set_xlabel('ica-one')
    ax.set_ylabel('ica-two')
    ax.set_zlabel('ica-three')
    plt.savefig(plot_path + '/ica_3D')
    

def rp_reconstruction_error(X_train, plot_path):
    reconstruction_errors = {}
    dimensions = list(range(2, X_train.shape[1], 1))
    print(dimensions)
    print(dimensions)
    for dimension in dimensions:
        rp = GaussianRandomProjection(n_components=dimension)
        x_prime = rp.fit_transform(X_train)
        A = np.linalg.pinv(rp.components_.T)
        reconstructed = np.dot(x_prime, A)
        rc_err = mean_squared_error(X_train, reconstructed)
        reconstruction_errors[dimension] = rc_err
    plt.figure()
    plt.plot(list(reconstruction_errors.keys()), list(reconstruction_errors.values()))
    plt.xlabel("Number of dimensions")
    plt.ylabel("RMSE")
    plt.title('Reconstruction Error')
    plt.savefig(plot_path + '/randomized_projection_re')
    plt.close()

def run_ICA(X,y,plot_path):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    #dims = list(np.arange(2,80,3))
    dims.append(X.shape[1])
    ica = ICA(random_state=1, max_iter=10)
    kurt = []

    for dim in dims:
        print(dim)
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis")
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.savefig(plot_path + '/ICA_DR')

def run_SVD(X_train, plot_path):
    reconstruction_errors = {}
    dimensions = list(range(2, X_train.shape[1], 1))
    print(dimensions)
    print(dimensions)
    for dimension in dimensions:
        svd = TruncatedSVD(n_components=dimension)
        x_prime = svd.fit_transform(X_train)
        A = np.linalg.pinv(svd.components_.T)
        reconstructed = np.dot(x_prime, A)
        rc_err = mean_squared_error(X_train, reconstructed)
        reconstruction_errors[dimension] = rc_err
    plt.figure()
    plt.plot(list(reconstruction_errors.keys()), list(reconstruction_errors.values()))
    plt.xlabel("Number of dimensions")
    plt.ylabel("RMSE")
    plt.title('Reconstruction Error')
    plt.savefig(plot_path + '/svd_re')
    plt.close()

def plotClusters(X_train, y_train, plot_path):
    df = pd.DataFrame()
    label_list = y_train.tolist()
    label_list = list(map(int, label_list))
    df['Component one'] = X_train[:, 0]
    df['Component two'] = X_train[:, 1]
    df['Component three'] = X_train[:, 2]
    ax = plt.figure().gca(projection='3d')
    ax.scatter(
        xs=df["Component one"],
        ys=df["Component two"],
        zs=df["Component three"],
        c=label_list)
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')
    plt.savefig(plot_path + '/3d_plot')





if __name__ == "__main__":
    '''
    Train algorithms and explore data here.
    '''
    ####################################################################################################################
    # Preprocessing and Data importing
    ####################################################################################################################
    # basically following these steps https://debuggercafe.com/image-classification-with-mnist-dataset/
    start = time.time()  # Time the entire runtime

    # Import the MNIST dataset, extract data and labels
    # mnist_data = import_data("mnist_784")
    # X_whole, y_whole = mnist_data['data'], mnist_data['target']
    # y_whole = y_whole.astype(np.uint8)
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/mnist_plots'


    # Import the spambase dataset, extract data and labels
    data = import_data("spambase")
    X_whole, y_whole = data['data'], data['target']
    y_whole = y_whole.astype(np.uint8)  # convert labels to integers to enable graphing easier
    plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots'

    X = X_whole[0::1]
    y = y_whole[0::1]

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # ## Part 1 ##
    # # Conduct K-means and EM analysis
    # kmeans_analysis(X_train, y_train, plot_path)
    # EM_analysis(X_train, y_train, plot_path)

    # ## Part 2 ##
    # # Conduct randomized projection reconstruction error
    # rp_reconstruction_error(X_train, plot_path)

    # # ICA Analysis
    # run_ICA(X_train,y_train,plot_path)

    # # SVD Analysis
    # run_SVD(X_train, plot_path)

    # # PCA Variance
    # pca(X_train, plot_path)

    # # PCA Graphical Analysis (initial)
    # pca_analysis(X_train, y_train, plot_path)

    # # ICA Graphical Analysis (initial)
    # ica_analysis(X_train, y_train, plot_path)


    ##### Part 3 ######
    # Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. 
    #Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method. 
    # You should look at all of them, but focus on the more interesting findings in your report.

    # ## For Spambase ##
    # # PCA - 3 components
    # # SVD - 2 components
    # # ICA - 3 components
    # # RP - 10 components (This will be adjusted later)

    # # PCA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots_part3_pca'

    # new_x_train = PCA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # plotClusters(pc_new_x_train, y_train, plot_path)

    # # SVD
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots_part3_svd'

    # new_x_train = TruncatedSVD(n_components=2)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['component 1', 'component 2'])

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # new_x_train = TruncatedSVD(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)


    # # ICA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots_part3_ica'

    # new_x_train = ICA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['component 1', 'component 2', 'component 3'])

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # plotClusters(pc_new_x_train, y_train, plot_path)

    # # RP
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots_part3_rp'

    # new_x_train = GaussianRandomProjection(n_components=10)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['component 1', 'component 2','component 3', 'component 4','component 5', 'component 6','component 7', 'component 8','component 9', 'component 10'])

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # new_x_train = ICA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)

    # ## For MNIST ##
    # # PCA - 200 components
    # # SVD - 200 components
    # # ICA - 700 components
    # # RP - 350 components (This will be adjusted later)

    # # PCA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/MNIST_plots_part3_pca'

    # new_x_train = PCA(n_components=200)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train)
    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # new_x_train = PCA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)

    # # SVD
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/MNIST_plots_part3_svd'

    # new_x_train = TruncatedSVD(n_components=200)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train)

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)

    # new_x_train = TruncatedSVD(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)


    # # ICA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/MNIST_plots_part3_ica'

    # new_x_train = ICA(n_components=700)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train)

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)
    # new_x_train = ICA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)

    # # RP
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/MNIST_plots_part3_rp'

    # new_x_train = GaussianRandomProjection(n_components=350)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train)

    # kmeans_analysis(new_x_train_Df, y_train, plot_path)
    # EM_analysis(new_x_train_Df, y_train, plot_path)

    # new_x_train = GaussianRandomProjection(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)
    # plotClusters(pc_new_x_train, y_train, plot_path)


    ##### Part 4 (Did this one for spambase...)#####

    def test_NN(X_train, y_train, plot_path):
    

        # Split the initial data
        xtrain , xtest ,ytrain, ytest = train_test_split(X,y,test_size =0.2,random_state =42)

        start=datetime.now()

        ### NNLearner Implementation ###
        nnlearner = nn.NNLearner(n_folds=3, verbose=True)  

        # Create a validation set - do another train/test split on the training data
        xtrain_val , xtest_val ,ytrain_val, ytest_val = train_test_split(X,y,test_size =0.2,random_state =42)

        # Initial Fit
        initial_classifier = MLPClassifier()
        initial_classifier.fit(xtrain_val, ytrain_val)

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        title = "Initial Learning Curves (Neural Nets)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        estimator = initial_classifier
        lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


        lc.savefig(plot_path + '/nn_learningcurve_initial.png')

        # Get a list of possible neural nets and their respective activation_types
        flag = 0
        clfs, activation_types = nnlearner.train(xtrain_val,ytrain_val,flag)
        # Get the nn that is correlated to the activation_type with highest accuracy
        momentum_values = "NA"
        learning_rates = "NA"
        hidden_layer_sizes_types = "NA"
        nn_choice_activation_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

        # Get a list of possible neural nets and their respective momentum values
        flag = 1
        clfs, momentum_values = nnlearner.train(xtrain_val,ytrain_val,flag)
        # Get the nn that is correlated to the momentum with highest accuracy
        activation_types = "NA"
        learning_rates = "NA"
        hidden_layer_sizes_types = "NA"
        nn_choice_momentum_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

        # Get a list of possible neural nets and their respective learning rates
        flag = 2
        clfs, learning_rates = nnlearner.train(xtrain_val,ytrain_val,flag)
        # Get the nn that is correlated to the learning rate with highest accuracy
        activation_types = "NA"
        momentum_values = "NA"
        hidden_layer_sizes_types = "NA"
        nn_choice_lr_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

        # Get a list of possible neural nets and their respective hidden_layer_sizes_types
        flag = 3
        clfs, hidden_layer_sizes_types = nnlearner.train(xtrain_val,ytrain_val,flag)
        # Get the decision tree that is correlated to the min_samples_split with highest accuracy
        activation_types = "NA"
        momentum_values = "NA"
        learning_rates = "NA"
        nn_choice_hiddenlayer_based = nnlearner.test(xtest_val,xtrain_val,ytest_val,ytrain_val,clfs,activation_types, momentum_values, learning_rates, hidden_layer_sizes_types, flag)

        # Now that we have the nn, time for tuning hyperparameters
        # Make a new classifier for this
        clf = MLPClassifier()
        clf.fit(xtrain_val, ytrain_val)
        best_params = nnlearner.tune_hyperparameters(clf, xtrain_val, ytrain_val)
        print("Best params are: ", best_params)

        # Now do one more fit based on best params above
        final_classifier = MLPClassifier(activation=best_params['activation'],hidden_layer_sizes=best_params['hidden_layer_sizes'], learning_rate=best_params['learning_rate'],momentum=best_params['momentum'],solver='sgd')
        final_classifier.fit(xtrain_val, ytrain_val)

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        title = "Learning Curves (Neural Nets)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        estimator = final_classifier
        lc = plot_learning_curve(estimator, title, xtrain_val, ytrain_val, cv=cv, n_jobs=-1)


        lc.savefig(plot_path + '/nn_learningcurve.png')

        # Now time for final accuracy score for test set
        nnlearner.final_test(final_classifier,xtest,ytest)

        print(datetime.now()-start)

    # # For PCA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pca'

    # new_x_train = PCA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

    # test_NN(new_x_train_Df, y_train, plot_path)

    # # For SVD
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_svd'

    # new_x_train = TruncatedSVD(n_components=2)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['principal component 1', 'principal component 2'])

    # test_NN(new_x_train_Df, y_train, plot_path)

    # # For ICA
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_ica'

    # new_x_train = ICA(n_components=3)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

    # test_NN(new_x_train_Df, y_train, plot_path)

    # # For RP
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_rp'

    # new_x_train = GaussianRandomProjection(n_components=10)
    # pc_new_x_train = new_x_train.fit_transform(X_train)

    # new_x_train_Df = pd.DataFrame(data = pc_new_x_train
    #             , columns = ['component 1', 'component 2','component 3', 'component 4','component 5', 'component 6','component 7', 'component 8','component 9', 'component 10'])

    # test_NN(new_x_train_Df, y_train, plot_path)
    


# ##### Part 5 (Did this one for Spambase) #####
# # For PCA
# new_x_train = PCA(n_components=3)
# pc_new_x_train = new_x_train.fit_transform(X_train)

# new_x_train_Df = pd.DataFrame(data = pc_new_x_train
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_pca_kmeans'
# kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(kmeans, y_train, plot_path)
# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_pca_em'
# gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(gmm, y_train, plot_path)

# # For SVD
# new_x_train = TruncatedSVD(n_components=2)
# pc_new_x_train = new_x_train.fit_transform(X_train)

# new_x_train_Df = pd.DataFrame(data = pc_new_x_train
#             , columns = ['principal component 1', 'principal component 2'])

# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_svd_kmeans'
# kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(kmeans, y_train, plot_path)
# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_svd_em'
# gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(gmm, y_train, plot_path)

# # For ICA
# new_x_train = ICA(n_components=3)
# pc_new_x_train = new_x_train.fit_transform(X_train)

# new_x_train_Df = pd.DataFrame(data = pc_new_x_train
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_ica_kmeans'
# kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(kmeans, y_train, plot_path)
# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_ica_em'
# gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(gmm, y_train, plot_path)

# # For RP
# new_x_train = GaussianRandomProjection(n_components=10)
# pc_new_x_train = new_x_train.fit_transform(X_train)

# new_x_train_Df = pd.DataFrame(data = pc_new_x_train
#             , columns = ['component 1', 'component 2','component 3', 'component 4','component 5', 'component 6','component 7', 'component 8','component 9', 'component 10'])

# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_rp_kmeans'
# kmeans = KMeans(n_clusters=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(kmeans, y_train, plot_path)
# plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_nn_pt5_rp_em'
# gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(pc_new_x_train)
# test_NN(gmm, y_train, plot_path)




    





