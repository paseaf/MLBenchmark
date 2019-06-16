# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:07:41 2019

@author: sechs
"""
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np

# def plot(metrics: [str] = ["Cross" for i in range(3)],
#          x_values: [[int]] = [[[i+j+0, i+j+1, i+j+2, i+j+3] for i in range(5)] for j in range(3)],
#          y_values: [[int]] = [[list(range(i*4+j, (i+1)*4+j)) for i in range(5)] for j in range(3)],
#          mls: [str] = ["boosting" for i in range(5)]
#          ):
#
#     fig, axs = plt.subplots(len(metrics), len(mls), figsize=(10, 5))
#     for z in range(len(mls)):
#         axs[0, z].set_title(mls[z])
#         for y in range(len(metrics)):
#             if y != fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0] - 1:
#                 axs[y, z].set_xticklabels([])
#             if z > 0 and z < fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]:
#                 axs[y, z].set_yticklabels([])
#             if z == fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1] - 1:
#                 axs[y, z].set_ylabel(metrics[y])
#                 #axs[y, z].yabel.tick_right
#                 #axs[y, z].yabel.set_label_position("right")
#
#     for x in range(len(x_values)):
#         for i in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]):
#             for j in range(fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]):
#                 axs[i, j].plot(x_values[x], y_values[x], ["b", "g", "r", "c", "m", "y", "k", "w"][x%8] + "o")
#
#     plt.ylabel(r'Classification Metric Value', fontsize = 25, horizontalalignment='center', verticalalignment='center', y = 1.5, labelpad = 480.0)
#     plt.xlabel("Number of variables for training", fontsize = 25, horizontalalignment='center', x = -1.5)
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.savefig('plot1.jpg')
#     plt.show()
#     return


def plot(points_dict: dict, acc_idx_list: list, train_method_list: list, valid_method: str):
    range_dict = {'ACC': (0.65, 1.05),
                  'BER': (0.65, 1.05),
                  'CE': (-0.03, 0.55),
                  'CRAMERV': (0, 1.05),
                  'KAPPA': (0, 1.05)}
    fig, axs = plt.subplots(len(acc_idx_list), len(train_method_list), figsize=(5, 5), gridspec_kw = {'wspace':0, 'hspace':0})
    for train_method_id in range(len(train_method_list)):
        axs[0, train_method_id].set_title(train_method_list[train_method_id])
        for acc_id in range(len(acc_idx_list)):
            if acc_id != fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0] - 1:
                axs[acc_id, train_method_id].set_xticklabels([])
            if train_method_id > 0 and train_method_id < fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]:
                axs[acc_id, train_method_id].set_yticklabels([])
            if train_method_id == fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1] - 1:
                axs[acc_id, train_method_id].yaxis.set_label_position("right")
                axs[acc_id, train_method_id].set_ylabel(acc_idx_list[acc_id])

    for acc_id in range(len(acc_idx_list)):
        for train_method_id in range(len(train_method_list)):
            axs[acc_id][train_method_id].grid('on', linestyle='--')     # print grid for each subplot
            axs[acc_id][train_method_id].set_ylim(range_dict[acc_idx_list[acc_id]])  # set y axis range for each acc idx
            axs[acc_id][train_method_id].scatter(*points_dict[(train_method_list[train_method_id], acc_idx_list[acc_id])],
                                                 s=10, c='r')

    fig.text(0.5, 0.008, 'Number of files for training', ha='center', fontsize=12)
    fig.text(0.001, 0.5, 'Classification Metric Value', va='center', rotation='vertical', fontsize=12)
    dt = datetime.datetime.now().replace(microsecond=0).strftime('%Y%m%d_%H%M%S')    # timestamp to name file
    plt.savefig(f'{dt}_{valid_method}.jpg')  # filename = [datetime]_[valid_method_name]
    plt.show()


if __name__ == "__main__":
    plot()
