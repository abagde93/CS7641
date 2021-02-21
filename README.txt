Code can be found here: https://github.com/abagde93/CS7641

All analysis was done using sklearn. The code is divided up into the follwing sections:

main.py --> This runs all the experiments for all the learners

Each of the learners have train() and test() methods, and also methods to find the best params from gridSearchCV, as well a final_test() method for getting the final accuracy:
DTLearner.py --> Decision Tree Implementation
KNNLearner.py --> K Nearest Neighbor Implementation
SVMLearner.py --> Support Vector Machine Implementation
NNLearner.py --> Neural Network Implementation
BoostingLearner.py --> Boosting Implementation

plot_learning_curve.py --> Plots learning curves when called 

****************************
Datasets used (OpenML built-ins):

mnist_784: https://www.openml.org/d/554
spambase: https://www.openml.org/d/44

******************************
How to run:

1. Navigate to main.py file
2. Locate the following line in get_data() --> "mnist_data = fetch_openml('mnist_784')". To get results for spambase change to 'fetch_openml('spambase')'
3. Locate following block in get_data():
    # Take a subset of the data (10%)
    X = X_whole[0::10]
    y = y_whole[0::10]

    This can be changed to whatever subset is preffered (for runtime purposes)

4. Locate following block at end of file:
   if __name__ == "__main__":  		 
    X_whole, y_whole, X, y = get_data() 	   		     		  		  		    	 		 		   		 		  
    test_DT(X_whole, y_whole, X, y)  
    test_NN(X_whole, y_whole, X, y)	
    test_SVM(X_whole, y_whole, X, y)
    test_KNN(X_whole, y_whole, X, y)
    test_Boosting(X_whole, y_whole, X, y)

    If only certain learners are desired to be run, the rest can be commented out.

To run code --> "python3 main.py"

This will generate plots in a seperate directory (might have to change paths depending on OS), and also Info files for each learner in the same dirctory as the code. These files contain accuracy rates,
hyperparameter optimizations, and runtimes


