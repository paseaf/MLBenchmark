from fileloader import FileLoader
import allModels as mlm
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score

file_path='/home/ole/Documents/Informatik/SS19/DBPRO/EuroSAT/2750'
jpeg_loader = FileLoader(root_path=file_path, files_per_class=300)
jpeg_loader.set_class_list(['AnnualCrop', 'Forest', 'Industrial'])
jpeg_loader.set_control_set(num_of_bands=3, is_random=False) 
jpeg_loader.set_training_subsets(num_of_subsets=30, max_percent=0.5)

train_set_x_2, train_set_y_2 = jpeg_loader.training_subsets[2]
train_set_x_5, train_set_y_5 = jpeg_loader.training_subsets[5]
control_set_x, control_set_y = jpeg_loader.control_set

# LDA
lda = mlm.modelLDA(train_set_x_5.T, train_set_y_5.reshape(train_set_y_5.size, ))
pred = lda.predict(train_set_x_2.T)
print(accuracy_score(train_set_y_2.reshape(45,1), pred))

# KNN
knn = mlm.modelKNN(train_set_x_5.T, train_set_y_5.reshape(train_set_y_5.size,))
pred = knn.predict(train_set_x_2.T)
print(accuracy_score(train_set_y_2.reshape(45,1), pred))

# forest
forest = mlm.modelForest(train_set_x_5.T, train_set_y_5.reshape(train_set_y_5.size,))
pred = forest.predict(train_set_x_2.T)
print(accuracy_score(train_set_y_2.reshape(45,1), pred))

# SVM
svm = mlm.modelSVM(train_set_x_5.T, train_set_y_5.reshape(train_set_y_5.size,))
pred = forest.predict(train_set_x_2.T)
print(accuracy_score(train_set_y_2.reshape(45,1), pred))
