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

    # # Plotting Inertia
    # # What is Inertia? - The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares
    # # Inertia can be recognized as a measure of how internally coherent clusters are.
    # # This is the elbow method of determining clusters
    # # https://predictivehacks.com/k-means-elbow-method-code-for-python/
    # inertia = {}
    # for cluster in clusters:
    #     kmeans = KMeans(n_clusters=cluster, max_iter=1000, random_state=4469).fit(x_train)
    #     inertia[cluster] = kmeans.inertia_
    # plt.figure()
    # plt.plot(list(inertia.keys()), list(inertia.values()))
    # plt.xlabel("Number of clusters")
    # plt.ylabel("Inertia")
    # plt.savefig(plot_path + '/clusters_vs_inertia')

    # Its hard to see an elbow in this plot, so lets look at other metrics
    # Specifically Silhoutte Coefficient, Homogeneity, and Completeness 

    # silhoutte_coefficients_scores = {}
    # homogeneity_scores = {}
    # completeness_scores = {}
    # for cluster in clusters:
    #     gmm = GaussianMixture(n_components=cluster, max_iter=1000, random_state=4469).fit(x_train)
    #     label = gmm.predict(x_train)
    #     silhoutte_coefficient = silhouette_score(x_train, label, metric='euclidean')
    #     silhoutte_coefficients_scores[cluster] = silhoutte_coefficient
    #     homo_score = homogeneity_score(y_train, label)
    #     homogeneity_scores[cluster] = homo_score
    #     comp_score = completeness_score(y_train, label)
    #     completeness_scores[cluster] = comp_score
    # plt.figure()
    # plt.plot(list(silhoutte_coefficients_scores.keys()), list(silhoutte_coefficients_scores.values()), label='silhoutte_coefficients_score')
    # plt.plot(list(homogeneity_scores.keys()), list(homogeneity_scores.values()), label='homogeneity_score')
    # plt.plot(list(completeness_scores.keys()), list(completeness_scores.values()), label='completeness_score')
    # plt.xlabel("Number of clusters")
    # plt.title("Clustering Performance Evaluation")
    # plt.legend()
    # plt.savefig(plot_path + '/clustering_performance_evaluation_EM')

    # From this we picked a cluster size of 45 as optimal for Spambase
    # Lets see the percentage of clusters that align with classes
    # 14.3% for Kmeans (not great)
    # TODO: PLay around with this and optimal cluster size
    gmm = GaussianMixture(n_components=10, max_iter=1000, random_state=4469).fit(x_train)
    print("Accuracy score is: ", accuracy_score(label, y_train))


 

def pca_analysis(x_train, y_train, plot_path):
    

    dimensions = list(range(2,100,1))
    for dimension in dimensions:

        # Perform PCA
        pca = PCA(n_components=dimension, random_state=4469)
        pca.fit(x_train)
        print(dimension, ": ", sum(pca.explained_variance_ratio_))

    # For MNIST (5% data) it's seen that 86 components capture 90% of the variance (just using 90% as a 'decent enough' benchmark)
    # We can even get variance per component
    comp = 86
    pca = PCA(n_components=comp, random_state=4469)
    pca_result = pca.fit_transform(x_train)

    x_train_reduced = {}
    for i in range(comp):
        field_name = "pca-"+str(i)
        x_train_reduced[field_name] = pca_result[:,i]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print(x_train_reduced)
    # rndperm = np.random.permutation(x_train.shape[0])
    
    # ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    # ax.scatter(
    #     xs=x_train.loc[rndperm,:]["pca-0"], 
    #     ys=x_train.loc[rndperm,:]["pca-1"], 
    #     zs=x_train.loc[rndperm,:]["pca-2"], 
    #     cmap='tab10'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    # plt.show()






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
    mnist_data = import_data("mnist_784")
    X_whole, y_whole = mnist_data['data'], mnist_data['target']
    y_whole = y_whole.astype(np.uint8)
    plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/mnist_plots'


    # # Import the spambase dataset, extract data and labels
    # data = import_data("spambase")
    # X_whole, y_whole = data['data'], data['target']
    # y_whole = y_whole.astype(np.uint8)  # convert labels to integers to enable graphing easier
    # plot_path = '/Users/ajinkya.bagde/Desktop/CS7641/AS3/Spambase_plots'

    # Take a subset of the data (10%)
    X = X_whole[0::20]
    y = y_whole[0::20]

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Conduct K-means analysis
    #kmeans_analysis(X_train, y_train, plot_path)
    #EM_analysis(X_train, y_train, plot_path)
    pca_analysis(X_train, y_train, plot_path)



