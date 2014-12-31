# quick and dirty script for playing with scikit-learn SVM

import sklearn.svm as skm
import numpy as np
import pandas as pd


# Read titanic data
data = pd.read_csv("./data/titanic/train.csv", index_col="PassengerId")




#------------------ Example from sklearn website -------------------
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# Sklearn SVM object for classification
svc = skm.SVC(kernel='rbf')

# Training data
#   input:  ndarray: nobs x nfeatures
#   target: ndarray: nobs,
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# Train the classifier
results = svc.fit(iris_X_train,iris_y_train)

# Predict test data with classifier
prediction = results.predict(iris_y_test)
