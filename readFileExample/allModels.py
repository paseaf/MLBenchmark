from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# input mylists = [X_test,X_train,Y_test,Y_train] 
# returns model for further processing
def modelKNN(mylists, neighbors = 5):
    clf = neighbors.KNeighborsClassifier(n_neighbors=neighbors)
    knn_fit = clf.fit(mylists[0], mylists[2])
    #clf.score(mylists[1], mylists[3])
    return knn_fit

