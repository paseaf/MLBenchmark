# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:07:41 2019

@author: sechs
"""
import matplotlib.pyplot as plt
import random
import numpy as np

def plot(metrics: [str] = ["Cross" for i in range(3)],
         x_values: [[int]] = [[[i+j+0, i+j+1, i+j+2, i+j+3] for i in range(5)] for j in range(3)],
         y_values: [[int]] = [[list(range(i*4+j, (i+1)*4+j)) for i in range(5)] for j in range(3)],
         mls: [str] = ["boosting" for i in range(5)]
         ):
    
    fig, axs = plt.subplots(len(metrics), len(mls), figsize=(10, 5))
    for z in range(len(mls)):
        axs[0, z].set_title(mls[z])
        for y in range(len(metrics)):
            if y != fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0] - 1:
                axs[y, z].set_xticklabels([])
            if z > 0 and z < fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]:
                axs[y, z].set_yticklabels([])
            
    for x in range(len(x_values)):
        for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]):
            for j in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]):
                axs[i, j].plot(x_values[x], y_values[x], ["b", "g", "r", "c", "m", "y", "k", "w"][x%8] + "o")
                
    plt.ylabel(r'Classification Metric Value', fontsize = 25, horizontalalignment='center', verticalalignment='center', y = 1.5, labelpad = 480.0)
    plt.xlabel("Number of variables for training", fontsize = 25, horizontalalignment='center', x = -1.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('plot1.jpg')
    plt.show()
    return
    
if __name__ == "__main__":
    plot()