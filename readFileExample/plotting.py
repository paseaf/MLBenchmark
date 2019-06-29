# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:07:41 2019

@author: sechs
"""
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np


# set font size
TITLE = 18
TEXT = 16


def plot_acc(points_dict: dict, acc_idx_list: list, train_method_list: list, valid_method: str):
    fig, axs = plt.subplots(len(acc_idx_list), len(train_method_list), figsize=(20, 10),
                            gridspec_kw={'wspace': 0, 'hspace': 0},
                            sharex='col', sharey='row')

    for acc_id in range(len(acc_idx_list)):
        for train_method_id in range(len(train_method_list)):
            if acc_id == 0:
                axs[acc_id][train_method_id].set_title(train_method_list[train_method_id], fontsize=TEXT)
            if train_method_id == len(train_method_list) - 1:
                axs[acc_id][train_method_id].set_ylabel(acc_idx_list[acc_id], fontsize=TEXT)
                axs[acc_id][train_method_id].yaxis.set_label_position("right")
            axs[acc_id][train_method_id].grid('on', linestyle='--')  # print grid for each subplot
            x, y = points_dict[(train_method_list[train_method_id], acc_idx_list[acc_id])]
            x = x[1::2]  # only plot every 2nd result
            y = y[1::2]  # only plot every 2nd result
            axs[acc_id][train_method_id].scatter(x, y, s=10, c='r')
            x_ticks = x[1::2]  # set x-axis ticks
            axs[acc_id][train_method_id].set_xticks(x_ticks)

    fig.text(0.5, 0.004, 'Number of files in training', ha='center', fontsize=13)
    fig.text(0.00, 0.5, 'Classification Metric Value', va='center', rotation='vertical', fontsize=14)
    dt = datetime.datetime.now().replace(microsecond=0).strftime('%Y%m%d_%H%M%S')  # timestamp to name file
    plt.savefig(f'{dt}_accuracy_result_{valid_method}.jpg')  # filename = [datetime]_[valid_method_name]
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
    dt = datetime.datetime.now().replace(microsecond=0).strftime('%Y%m%d_%H%M%S')  # timestamp to name file
    plt.savefig(f'{dt}_time_result.jpg')  # filename = [datetime]
    plt.show()


if __name__ == "__main__":
    plot_acc()
