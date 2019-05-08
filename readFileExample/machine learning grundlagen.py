# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:29:06 2019

@author: sechs
"""
import numpy as np
#import lib
import matplotlib as mpl
import os
from sklearn.model_selection import train_test_split
import gdal

def load_images(path: str, file_ending: str=".jpg") -> (np.ndarray, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions

    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()

    A = sorted(os.listdir(path))
    for name in A:
        if (path + "/" + name).endswith(file_ending) :
            images += [np.asarray(mpl.image.imread(path + "/" + name), dtype = np.float64).flatten()]
    print( path.split("/")[-1] + " enthält ",len(images) , " Bilder.")
    # TODO set dimensions according to first image in images
    dimension_y = images[0].shape[0]
    #dimension_x = images[0].shape[1]
    #return images, dimension_x, dimension_y
    return np.asarray(images), dimension_y

def load_image_folder(path:str, file_ending: str=".jpg",) -> [(str, np.ndarray, int)]:
    classes = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            x, y = load_images(root + "/" + name, file_ending)
            classes.append((name, x, y))
    return classes

if __name__ == '__main__' :
    #print(load_image_folder("./../Users/sechs/Downloads/EuroSAT/2750"))
    test = load_image_folder("C:/Users/sechs/Downloads/EuroSAT/2750")
    #test2 = test[0][1]
    #print(type(test[0]))
    #print(type(test[0][1]))
    #print(type(test[0][1][0]))
    #print(type(test[0][1][0][0]))
    #print(np.asarray(test[0][1]).shape)
    
    X_train, X_test, y_train, y_test = [],[],[],[]
    mylists = [X_train, X_test, y_train, y_test]
    for i in range(len(test)):
        for x, lst in zip(train_test_split(test[i][1], np.repeat(i, test[i][1].shape[0]),test_size = 0.8), mylists):
            lst.append(x)
            
    for i in range(len(mylists)):
        mylists.append(np.concatenate(mylists[0], axis = 0))
        mylists.pop(0)
    print(mylists[0].shape)
    """
    checklist = np.zeros(len(test))
    print(len(y_train))
    for i in range(len(y_train)):
        checklist[y_train[i]] += 1
    for i in range(len(checklist)):
        print(test[i][0], ": ", checklist[i])
    """
    #  X_train, X_test, y_train, y_test += train_test_split(np.asarray(test[i][1]), np.repeat(i, len(test[i,1])),test_size = 0.7)
    #all(setting[1][0].shape == test[0][1][0].shape for setting in test)