# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:07:41 2019

@author: sechs
"""
import matplotlib.pyplot as plt
    
def plot(point_list):
    """point_list = []
    dictionary = dict()
    for x in point_list:
        if dictionary.get((x.train_method_name, x.acc_idx_name) is None):
            dictionary.update({(x.train_method_name, x.acc_idx_name) : [x]})
        else:
           dictionary.update({(x.train_method_name, x.acc_idx_name) : dictionary.get((x.train_method_name, x.acc_idx_name)).append(x)}) 
    print("Hello, World")
    for y in dictionary.values:
        plt.plot([x.num_of_files for x in y])
    """
    plt.plot([1, 2, 5])
    plt.ylabel('Classification Metric Value')
    plt.show()
    
if __name__ == "__main__":
    plot(1)