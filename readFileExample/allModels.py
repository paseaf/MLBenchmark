""" 
input: mylists = [X_test,X_train,Y_test,Y_train]
returns: model for each function
"""
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.svm import SVC as svm
from sklearn.ensemble import RandomForestClassifier as randomForest
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.tree import DecisionTreeClassifier as ctree
from sklearn.ensemble import GradientBoostingClassifier as gdb
from sklearn.linear_model import LogisticRegression as lr

# Linear Discriminant Analysis
def modelLDA(mylists):
    model = lda()
    fit = model.fit(mylists[0], mylists[2])
    return fit

# returns model for further processing
def modelKNN(mylists, neighbors = 5):
    model = knn(n_neighbors=neighbors)
    fit = model.fit(mylists[0], mylists[2])
    #fit.score(mylists[1], mylists[3])
    return fit

# random forests
def modelForest(mylists, trees = 100):
    model = randomForest(n_estimators=trees)
    fit = model.fit(mylists[0], mylists[2])
    return fit

# support vector machine
def modelSVM(mylists):
    model = svm()
    fit = model.fit(mylists[0], mylists[2])
    return fit

# multilayered perceptron
def modelMLP(mylists):
    model = mlp()
    fit = model.fit(mylists[0], mylists[2])
    return fit

# multilayered perceptron ensemble
def modelMLPE(mylists):
    """ model = randomForest(n_estimators=n_trees)
    fit = model.fit(mylists[0], mylists[2])
    return fit """

# ctree
def modelCT(mylists):
    model = ctree()
    fit = model.fit(mylists[0], mylists[2])
    return fit

# boosting
def modelBoost(mylists):
    model = gdb()
    fit = model.fit(mylists[0], mylists[2])
    return fit

# logistic regression
def modelLR(mylists):
    model = lr()
    fit = model.fit(mylists[0], mylists[2])
    return fit