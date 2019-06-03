# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:07:41 2019

@author: sechs
"""
import matplotlib.pyplot as plt

def plot(ax, x_values: [int], y_values: [int], methode: str ):
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
    ax.plot(x_values, y_values, "ro")
    ax.set_title(methode)
    return
    
if __name__ == "__main__":
    
    fig, axs = plt.subplots(3, 5, figsize=(8, 4.8))
    x_values = [[0, 1, 2, 3] for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1])]
    y_values = [list(range(i*4, (i+1)*4)) for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1])]
    mls = ["boosting" for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1])]
    print(fig.axes[0].get_subplotspec().get_gridspec().get_geometry())
    for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]):
        for j in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]):
            plot(axs[i, j], x_values, y_values, mls[j] if i == 0 else "")
            if i != fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0] - 1:
                axs[i, j].get_xaxis().set_visible(False)
            if j != 0:
                axs[i, j].get_yaxis().set_visible(False)
                
    plt.ylabel("Classification Metric Value", fontsize = 100)
    plt.xlabel("Number of variables for training")
    plt.subplots_adjust(wspace=0, hspace=0)  
    #plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.savefig('plot1.jpg')
    plt.show()
    