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


def plot_acc(points_dict: dict, acc_idx_list: list, train_method_list: list, valid_method: str):
    range_dict = {'ACC': (-0.08, 1.08),
                  'BER': (-0.08, 1.08),
                  'CE': (-0.08, 1.08),
                  'CRAMERV': (-0.08, 1.08),
                  'KAPPA': (-0.08, 1.08)}
    fig, axs = plt.subplots(len(acc_idx_list), len(train_method_list), figsize=(10, 5), gridspec_kw = {'wspace':0, 'hspace':0})
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

    fig.text(0.5, 0.008, 'Number of files in training', ha='center', fontsize=12)
    fig.text(0.001, 0.5, 'Classification Metric Value', va='center', rotation='vertical', fontsize=12)
    dt = datetime.datetime.now().replace(microsecond=0).strftime('%Y%m%d_%H%M%S')    # timestamp to name file
    plt.savefig(f'{dt}_{valid_method}.jpg')  # filename = [datetime]_[valid_method_name]
    plt.show()

def plot_time(time_results: dict, train_method_list: list):
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/markevery_prop_cycle.html
    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']

    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    lines = []  # store one line for each mlm for later labeling
    # set font size
    TITLE = 18
    TEXT = 16

    for pid, key in enumerate(time_results):  # draw two subplots respectively
        axs[pid].set_title(key, fontsize=TITLE)  # set subplot title
        axs[pid].grid('on', linestyle='--')     # print grid for each subplot
        axs[pid].tick_params(axis='both', which='major', labelsize=TEXT)  # set tick size
        for mid, train_method in enumerate(train_method_list):  # plot for each training method
            x, y = time_results[key][train_method]
            axs[pid].scatter(x, y, s=10, c=colors[mid])  # plot scatter
            fit = np.polyfit(x, y, 1)
            fit_fn = np.poly1d(fit)
            # axs[pid].plot(x, y, 'o', x, fit_fn(x), '-', )
            line = axs[pid].plot(x, fit_fn(x), color=colors[mid])
            if pid == 0:
                lines += line  # add line to list for later plotting
            # axs[pid].xlim(x[0], x[-1])  # buggy because x is not sorted

    axs[1].set_xlabel('Number of files in training', fontsize=TITLE)  # set x label for the lower graph
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc="center left", borderaxespad=0)
    fig.legend(lines, train_method_list, 'center right',  fontsize=TEXT, title="MLM", title_fontsize=TITLE)

    plt.gcf().subplots_adjust(left=0.1)
    fig.text(0.01, 0.5, 'Time (s)', fontsize=TITLE, ha='center', va='center', rotation='vertical')
    plt.show()


if __name__ == "__main__":
    plot_acc()
