#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Dec 17 2021.

@author: Nassir, Mohammad
"""

from neural_network import Neural_network
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def produce_report(labels, true_labels, data, scores, save_path=None):
    """Produce graphs and AUC score report.

    Parameters
    ----------
    labels : binary numpy array
        labels indicating the binary classification decision made.
    true_labels : binary numpy array
        true labels indicating the binary classification of data.
    data : numpy array
        input data used as input to algorithms.
    scores : numpy array
        scores output by anomaly detector for each observation.
    save_path : string, optional
        Path to save the output graphs to. The default is None.

    Returns
    -------
    None.

    """
    tp = data[(labels == 1) & np.squeeze(true_labels == 1)]
    fp = data[(labels == 1) & np.squeeze(true_labels == 0)]
    tn = data[(labels == 0) & np.squeeze(true_labels == 0)]
    fn = data[(labels == 0) & np.squeeze(true_labels == 1)]

    print("sum: {}".format(len(fp)+len(fn)+len(tp)+len(tn)))

    # plot fp, fn, tp, tn
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(x=tp[:, 0], y=tp[:, 1],
                    color='darkorange',
                    label='Anomaly (True positive)')

    sns.scatterplot(x=fp[:, 0], y=fp[:, 1],
                    color='purple',
                    label='Normal (False positive)', marker='x')

    sns.scatterplot(x=tn[:, 0], y=tn[:, 1], zorder=0, alpha=0.3,
                    label='Normal (True negative)')

    sns.scatterplot(x=fn[:, 0], y=fn[:, 1], color='black',
                    label='Anomaly (False negative)', marker=',')

    # ax.set_title("Our model (validation data only)");
    plt.legend()

    if save_path:
        # save the path to where the paper figures are required
        plt.savefig(save_path)

    plt.show()

    # print the performance reports
    print(classification_report(true_labels, labels,
          target_names=['normal', 'abnormal'], output_dict=False))

    auc_score = roc_auc_score(true_labels, scores)
    print('AUC score: {}'.format(auc_score))


# %% accuracy (AUC score) over n_neurons + plot
def report_acc_over_neurons(train_data, predict_data, true_labels,
                            plot_title, x_axis_label, y_axis_label,
                            n_runs=1, n_neurons=512,
                            min_c=10, max_c=1000, save_path=None):
    """Plot AUC-score over number of neurons in network.

    Parameters
    ----------
    train_data : numpy array
        Data to train network upon.
    predict_data : numpy array
        Data to perform prediction upon.
    true_labels: binary numpy array
        True label of each predict_data observation.
    plot_title : string
        Title of plot.
    x_axis_label : string
        x-axis label.
    y_axis_label : string
        y-axis label.
    n_runs : integer, optional
        Number of time to run each network to take average results.
        The default is 5.
    n_neurons : int, optional
        Maximum number of neurons to use in the neural network.
        The default is 250.
    min_c : integer, optional
        Minimum number of samples taken by a neuron. The default is 25.
    max_c : integer, optional
        Maximum number of samples taken by a neuron. The default is 1000.
    save_path : string, optional
        Path to save plots to. The default is None.

    Returns
    -------
    None.

    """
    auc_scores = []

    for j in range(1, n_neurons):

        temp_auc_score = []

        # average auc_score over n_runs
        for _ in range(n_runs):
            net = Neural_network(n0=len(train_data), n1=j,
                                 min_n_connections=min_c,
                                 max_n_connections=max_c)
            net.fit(train_data)
            net.predict(predict_data)

            ts = roc_auc_score(true_labels, net.scores_)
            temp_auc_score.append(ts)

        mean_score = np.mean(temp_auc_score)
        auc_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = ax.plot(range(1, n_neurons), auc_scores)

    # Add labels
    plt.title(plot_title)
    # plt.ylabel(x_axis_label)
    # plt.xlabel(y_axis_label)
    plt.xlabel(x_axis_label, fontsize=25)
    plt.ylabel(y_axis_label, fontsize=25)
    plt.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path)

    plt.show()


# %% accuracy (AUC score) over n_subsamples + plot
def report_acc_over_subsamples(train_data, predict_data, true_labels,
                               plot_title, x_axis_label, y_axis_label,
                               n_runs=1, n_neurons=256, n_connections=200,
                               min_c=10, max_c=1000, save_path=None):
    """Plot AUC score of varying neuron fixed subsample size.

    Parameters
    ----------
    train_data : numpy array
        Data to train network upon.
    predict_data : numpy array
        Data to perform prediction upon.
    true_labels: binary numpy array
        True label of each predict_data observation.
    plot_title : string
        Title of plot.
    x_axis_label : string
        x-axis label.
    y_axis_label : string
        y-axis label.
    n_runs : integer, optional
        Number of time to run each network to take average results.
        The default is 5.
    n_neurons : int, optional
        Maximum number of neurons to use in the neural network.
        The default is 250.
    n_connections : integer, optional
        Number of connections per neuron two input/output layers.
        The default is 200.
    min_c : integer, optional
        Minimum number of samples taken by a neuron. The default is 25.
    max_c : integer, optional
        Maximum number of samples taken by a neuron. The default is 1000.
    save_path : string, optional
        Path to save plots to. The default is None.

    Returns
    -------
    None.

    """
    auc_scores = []

    for j in range(1, n_connections):  # not quite correct graph here

        temp_auc_score = []

        # average auc_score over n_runs
        for _ in range(n_runs):
            net = Neural_network(
                n0=len(train_data), n1=n_neurons,
                min_n_connections=min_c, max_n_connections=max_c,
                fixed_conn_per_l1_neuron=j+3)

            net.fit(train_data)
            net.predict(predict_data)

            ts = roc_auc_score(true_labels, net.scores_)
            temp_auc_score.append(ts)

        mean_score = np.mean(temp_auc_score)
        auc_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = ax.plot(range(1, n_connections), auc_scores)

    # Add labels
    plt.title(plot_title)
    plt.xlabel(x_axis_label, fontsize=25)
    plt.ylabel(y_axis_label, fontsize=25)
    plt.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path)

    plt.show()

# %% accuracy (AUC score) over n_neurons + plot


def report_f1_over_neurons(train_data, predict_data, true_labels,
                           plot_title, x_axis_label, y_axis_label,
                           n_runs=1, n_neurons=256,
                           min_c=10, max_c=1000, save_path=None):
    """Plot AUC-score over number of neurons in network.

    Parameters
    ----------
    train_data : numpy array
        Data to train network upon.
    predict_data : numpy array
        Data to perform prediction upon.
    true_labels: binary numpy array
        True label of each predict_data observation.
    plot_title : string
        Title of plot.
    x_axis_label : string
        x-axis label.
    y_axis_label : string
        y-axis label.
    n_runs : integer, optional
        Number of time to run each network to take average results.
        The default is 5.
    n_neurons : int, optional
        Maximum number of neurons to use in the neural network.
        The default is 250.
    min_c : integer, optional
        Minimum number of samples taken by a neuron. The default is 25.
    max_c : integer, optional
        Maximum number of samples taken by a neuron. The default is 1000.
    save_path : string, optional
        Path to save plots to. The default is None.

    Returns
    -------
    None.

    """
    f1_scores = []

    for j in range(1, n_neurons):

        temp_f1_score = []

        # average auc_score over n_runs
        for _ in range(n_runs):
            net = Neural_network(n0=len(train_data), n1=j,
                                 min_n_connections=min_c,
                                 max_n_connections=max_c)
            net.fit(train_data)
            net.predict(predict_data)

            cls_report_dic = classification_report(
                net.labels_, true_labels,
                target_names=['normal', 'abnormal'], output_dict=True)

            ts = cls_report_dic['abnormal']['f1-score']
            temp_f1_score.append(ts)

        mean_score = np.mean(temp_f1_score)
        f1_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = ax.plot(range(1, n_neurons), f1_scores)

    # Add labels
    plt.title(plot_title)
    plt.xlabel(x_axis_label, fontsize=25)
    plt.ylabel(y_axis_label, fontsize=25)
    plt.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path)

    plt.show()


# %% accuracy (AUC score) over n_subsamples + plot
def report_f1_over_subsamples(train_data, predict_data, true_labels,
                              plot_title, x_axis_label, y_axis_label,
                              n_runs=1, n_neurons=256, n_connections=200,
                              min_c=10, max_c=1000, save_path=None):
    """Plot AUC score of varying neuron fixed subsample size.

    Parameters
    ----------
    train_data : numpy array
        Data to train network upon.
    predict_data : numpy array
        Data to perform prediction upon.
    true_labels: binary numpy array
        True label of each predict_data observation.
    plot_title : string
        Title of plot.
    x_axis_label : string
        x-axis label.
    y_axis_label : string
        y-axis label.
    n_runs : integer, optional
        Number of time to run each network to take average results.
        The default is 5.
    n_neurons : int, optional
        Maximum number of neurons to use in the neural network.
        The default is 250.
    n_connections : integer, optional
        Number of connections per neuron two input/output layers.
        The default is 200.
    min_c : integer, optional
        Minimum number of samples taken by a neuron. The default is 25.
    max_c : integer, optional
        Maximum number of samples taken by a neuron. The default is 1000.
    save_path : string, optional
        Path to save plots to. The default is None.

    Returns
    -------
    None.

    """
    f1_scores = []

    for j in range(1, n_connections):

        temp_f1_score = []

        # average auc_score over n_runs
        for _ in range(n_runs):
            net = Neural_network(
                n0=len(train_data), n1=n_neurons,
                min_n_connections=min_c, max_n_connections=max_c,
                fixed_conn_per_l1_neuron=j+3)

            net.fit(train_data)
            net.predict(predict_data)

            cls_report_dic = classification_report(
                net.labels_, true_labels,
                target_names=['normal', 'abnormal'], output_dict=True)

            ts = cls_report_dic['abnormal']['f1-score']
            temp_f1_score.append(ts)

        mean_score = np.mean(temp_f1_score)
        f1_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = ax.plot(range(1, n_connections), f1_scores)

    # Add labels
    plt.title(plot_title)
    plt.xlabel(x_axis_label, fontsize=25)
    plt.ylabel(y_axis_label, fontsize=25)
    plt.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path)

    plt.show()
