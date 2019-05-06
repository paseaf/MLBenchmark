# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:29:06 2019

@author: sechs
"""
import numpy as np
#import lib
import matplotlib as mpl
import os
import sklearn
import gdal

def load_images(path: str, file_ending: str=".jpg") -> (list, int, int):
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
    dimension_x = images[0].shape[1]
    return images, dimension_x, dimension_y

def load_image_folder(path:str, file_ending: str=".jpg",) -> [(str, list, int, int)]:
    classes = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            classes.append((name, load_images(root + "/" + name, file_ending)))
    return classes

if __name__ == '__main__' :
    #print(load_image_folder("./../Users/sechs/Downloads/EuroSAT/2750"))
    test = load_image_folder("C:/Users/sechs/Downloads/EuroSAT/2750")
    print(type(test[0][1][0][0]))
    X_train, X_test, y_train, y_test = ([],[],[],[])
    #for i in range(len(test)):
    #  X_train, X_test, y_train, y_test += train_test_split(np.asarray(test[i][1]), np.repeat(i, len(test[i,1])),test_size = 0.7)
    #all(setting[1][0].shape == test[0][1][0].shape for setting in test)