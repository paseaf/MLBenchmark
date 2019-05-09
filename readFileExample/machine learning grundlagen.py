# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:29:06 2019

@author: sechs
"""
import os
import math
import random
import gdal
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split

"""
divide_in_train_test_data divides given images into train and test data.
The y values are the indices of the np.array in the given list.
The given data will be evenly represented in the train and test data (by percentage).

Arguments:
    data: data is a list containing tuples of name of the class and the given data (for example images) flattend
          so that one row represents one element of the data to learn from.
    
    testsize: testsize determines the procentual size of data used for testing afterwards.
    
    Return:
        X_train: list of data to train (each image as numpy.ndarray)
        X_test:list of data to test (each image as numpy.ndarray)
        y_train: list of data to train (each image as numpy.ndarray)
        y_test: list of data to test (each image as numpy.ndarray)
        
"""

def divide_in_train_test_data(data: [(str, np.ndarray)], testsize: float = 0.8)->(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    #calculating testsize from a given number (int or float) into a float in]0,1]
    testsize = testsize / float(10**(math.ceil(math.log10(testsize))))
    
    X_train, X_test, y_train, y_test = [],[],[],[]
    mylists = [X_train, X_test, y_train, y_test]
    
    #splitting every class data into X_train, X_test, y_train, y_test and ad it up in the lists
    for i in range(len(data)):
        for x, lst in zip(train_test_split(data[i][1], np.repeat(i, data[i][1].shape[0]),test_size = testsize), mylists):
            lst.append(x)
            
    #concatenate the test and train data sets into one np.array
    for i in range(len(mylists)):
        mylists.append(np.concatenate(mylists[0], axis = 0))
        mylists.pop(0)
    
    #returning the list as a tuple
    return tuple(mylists)
        
def load_images(path: str, file_ending: str=".jpg", percentage: float = 1.0, typ: type = np.float64) -> (np.ndarray):
    """
    Load a percentage of all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions

    file_ending: string that image files have to end with, if not->ignore file

    typ: typ detemines the type the images will be loaded as.

    Return:
    images: list of images (each image as numpy.ndarray and dtype decided by argument)
    """
    #normalization of the percentage for further use
    percentage = percentage / float(10**(math.ceil(math.log10(percentage))))
    
    images = []
    data = list(filter(lambda x: x.endswith(file_ending), os.listdir(path))) #all data with given file_ending found in the file
    count = math.ceil(len(data)*percentage) #number of images wanted

    #add up of the images
    for name in data[:count] :
        images += [np.asarray(mpl.image.imread(path + "/" + name), dtype = typ).flatten()]
    print( path.split("/")[-1] + " enthält ",len(images) , " Bilder.")
    return np.asarray(images)

def load_image_folder(path:str, file_ending: str=".jpg", percentage: float = 1.0, typ: type = np.float64) -> [(str, np.ndarray)]:
    """
    Loads a percentage of all images of every subfolder (non recursiv) with matplotlib that have given file_ending

    Arguments:
    path: path of directory of which contains folders containing image files that can be assumed to have all the same dimensions

    file_ending: string that image files have to end with, if not->ignore file

    typ: typ detemines the type the images will be loaded as.

    Return:
    classes: list of tuple containing the name of the folder and a matrix of flattened images (each image as numpy.ndarray and dtype decided by argument)
    """
    classes = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            x = load_images(root + "/" + name, file_ending, percentage, typ)
            classes.append((name, x))
    return classes

def  select_random_images(path: str, file_ending: str=".jpg", percentage: float = 1.0, typ: type = np.float64) -> (np.ndarray):
    "see load_image but random"
    percentage = percentage / float(10**(math.ceil(math.log10(percentage))))
    
    images = []
    data = list(filter(lambda x: x.endswith(file_ending), os.listdir(path))) #all data with given file_ending found in the file
    count = math.ceil(len(data)*percentage) #number of images wanted
    random.shuffle(data)

    #add up of the images
    for name in data[:count] :
        images += [np.asarray(mpl.image.imread(path + "/" + name), dtype = typ).flatten()]
    print( path.split("/")[-1] + " enthält ",len(images) , " Bilder.")
    return np.asarray(images)
    
def random_images(path:str, file_ending: str=".jpg", percentage: float = 1.0, typ: type = np.float64) -> (np.ndarray, [str]):
    """
    Loads a percentage of all images randomly choosen out of every subfolder (non recursiv) with matplotlib that have given file_ending.

    Arguments:
    path: path of directory of which contains folders containing image files that can be assumed to have all the same dimensions

    file_ending: string that image files have to end with, if not->ignore file

    typ: typ detemines the type the images will be loaded as.

    Return:
    data: randomly choosen but even spaced between the subfolders flattened images in a matrix (each image as numpy.ndarray and dtype decided by argument)
    classes: list of tuple including the name of every subfolder with the number of pictures taken.
    """
    classes, data = [], []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            x = select_random_images(root + "/" + name, file_ending, percentage, typ)
            classes.append((name, len(x)))
            data.append(x)
    data = np.concatenate(data, axis = 0)
    return data, classes
    

if __name__ == '__main__' :
#    randomclasses = load_image_folder("C:/Users/sechs/Downloads/EuroSAT/2750", percentage = 100, typ = np.uint8)
    randomclasses = random_images("C:/Users/sechs/Downloads/EuroSAT/2750", percentage = 20, typ = np.uint8)
    print(randomclasses[0].shape)
    #X_train, X_test, y_train, y_test = divide_in_train_test_data(randomclasses, 0.8)
    #mylists = [X_train, X_test, y_train, y_test]
    #for i in mylists:
    #   print(i.shape)
    
