"""
"""
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

"""  
"""
def modelLDA(train_set_x, train_set_y):
    model = LinearDiscriminantAnalysis()
    fit = model.fit(train_set_x, train_set_y)
    return fit

"""  
determine optimal choice of neighbors
weight by distance
"""
def modelKNN(train_set_x, train_set_y, n_neighbors=None):
    # optimize for optimal number neighbors or take nr of neighbors
    # if n_neighbors is None:
        # param_grid = {'n_neighbors': range(1, int(np.sqrt(train_set_y.shape[0])))}
        # grid = GridSearchCV(estimator=KNeighborsClassifier(weights='distance'), param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        # fit = grid.fit(train_set_x, train_set_y)
        # print(f"optimal hyper-parameters: {fit.best_params_}")
    # else:
    n_neighbors = int(np.sqrt(train_set_y.shape[0]))    # commented out gridsearch for
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    fit = knn.fit(train_set_x, train_set_y)
    return fit

"""
RandomForest does not overfit by number of trees
max_depth: The maximum depth of the tree
min_samples_split: The minimum number of samples required to split an internal node
"""
def modelForest(train_set_x, train_set_y, trees = 10):
    param_grid = {'max_depth': range(1,20), 'min_samples_split': range(2,20)}
    # use 'area under the curve' as optimization value
    #grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=trees), param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    # use 'accuracy' as optimization value
    # grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=trees), param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    # fit = grid.fit(train_set_x, train_set_y)
    # print(f"optimal hyper-parameters: {fit.best_params_}")

    # fixed parameter
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, bootstrap=True,
                                   max_features='sqrt', oob_score=True)
    model.fit(train_set_x, train_set_y)

    return model

"""  
use Radial Basis Function as kernel and then GridSearch to find optimal 'gamma' and 'C' parameters
"""
def modelSVM(train_set_x, train_set_y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = {'gamma':gamma_range, 'C': C_range}
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    model = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv, n_jobs=-1)
    fit = model.fit(train_set_x, train_set_y)
    print(f"optimal hyper-parameters: {fit.best_params_}")
    return fit

"""  
nr of layers?: start with few hidden neurons and few hidden layers
scale?: scale between [0,1] or mean = 0 & var = 1
"""
def modelMLP(train_set_x, train_set_y):
    #param_grid = {'solver': ['lbfgs','adam'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
    # param_grid = {'solver': ['lbfgs','adam'], 'hidden_layer_sizes':np.arange(1, 10)}
    # model = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    # fit = model.fit(train_set_x, train_set_y)
    # return fit
    m, N = 1, train_set_y.shape[0]
    model = MLPClassifier(hidden_layer_sizes=(int(2*np.sqrt((m+2)*N)), int(m*np.sqrt(N/(m+2)))))
    # model = MLPClassifier(hidden_layer_sizes=(5,))
    model.fit(train_set_x, train_set_y)
    return model


"""  
TODO: implement ensemble method
"""
def modelMLPE(train_set_x, train_set_y):
    mlp_1 = MLPClassifier(hidden_layer_sizes=(5,))
    mlp_2 = MLPClassifier(hidden_layer_sizes=(10,))
    mlp_3 = MLPClassifier(hidden_layer_sizes=(15,))
    ensemble = VotingClassifier(estimators=[('mlp_1',mlp_1), ('mlp_2',mlp_2), ('mlp_3',mlp_3)])
    model = ensemble.fit(train_set_x, train_set_y)
    return model

"""  
CTree
max_depth: The maximum depth of the tree
min_samples_split: The minimum number of samples required to split an internal node (default=2)
min_samples_leaf: The minimum number of samples required to be at a leaf node (deafult=1)
"""
def modelCT(train_set_x, train_set_y):
    # ctree = DecisionTreeClassifier(random_state=1)
    # param_grid = {'max_depth':range(1,50), 'min_samples_split':range(1,50)}
    # grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
    # fit = grid.fit(train_set_x, train_set_y)
    # print(f"optimal hyper-parameters: {fit.best_params_}")
    # return fit
    model = DecisionTreeClassifier().fit(train_set_x, train_set_y)
    return model

"""  
Boosting
deicision tree as weak learner by default
"""
def modelBoost(train_set_x, train_set_y):
    model = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
    fit = model.fit(train_set_x, train_set_y)
    return fit

"""
Logistic Regression
C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
"""
def modelLR(train_set_x, train_set_y):
    # logreg = LogisticRegression()
    # param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    # kfold = KFold(n_splits=5, random_state=7)
    # grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=kfold)
    # fit = grid.fit(train_set_x, train_set_y)
    # print(f"optimal hyper-parameters: {fit.best_params_}")
    # return fit
    model = LogisticRegression(multi_class='ovr', solver='saga', max_iter=200, n_jobs=-1)
    model.fit(train_set_x, train_set_y)
    return model