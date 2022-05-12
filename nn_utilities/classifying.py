#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Feb  8 13:35:48 2022.

@author: nassirmohammad
"""

from scipy.io import loadmat
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import time
import os

# the below are used via eval()
from perception import Perception  # NOQA
from neural_network import Neural_network  # NOQA
from pyod.models.hbos import HBOS  # NOQA
from pyod.models.iforest import IForest  # NOQA
from sklearn.ensemble import IsolationForest  # NOQA
from pyod.models.knn import KNN  # NOQA
from pyod.models.lof import LOF  # NOQA
from pyod.models.mcd import MCD  # NOQA
from pyod.models.ocsvm import OCSVM  # NOQA
from pyod.models.pca import PCA  # NOQA
from sklearn.cluster import DBSCAN  # NOQA
import warnings  # NOQA

# %% get file data


def get_file_data(base_path, file_name):

    # join paths to get file to path
    file_path = os.path.join(base_path, file_name)

    if file_name.endswith((".mat")):

        dataset_name = os.path.splitext(file_name)[0]

        # read in the file into numpy ndarray
        data = loadmat(file_path)

        # get the data and show
        X_original = data['X']
        y = data['y']

    elif file_name.endswith((".csv")):

        dataset_name = os.path.splitext(file_name)[0]

        df_temp = pd.read_csv(file_path, header=None)

        # replace label column with 0's and 1's
        df_temp.iloc[:, -1] = np.where(df_temp.iloc[:, -1] == 'n', 0, 1)

        # read data in as numpy arrays
        X_original = df_temp.iloc[:, 0:-1].values

        y = df_temp.iloc[:, -1].values
        y = y.reshape(y.shape[0], -1)

    elif file_name.endswith((".matv7")):

        dataset_name = os.path.splitext(file_name)[0]

        arrays = {}
        f = h5py.File(file_path)
        for k, v in f.items():
            arrays[k] = np.array(v)

        X_original = arrays['X'].T
        y = arrays['y'].T

    else:
        # do nothing as file formatted not supported
        dataset_name = None
        X_original = None
        y = None

    return dataset_name, X_original, y

# %% apply anomaly detection algorithms to multidimensional data


def call_init_classifier(clf_name, data):

    start = time.perf_counter()

    if clf_name == 'NeuralNetwork':
        result = Neural_network(n0=len(data),
                                n1=256,
                                min_n_connections=10,
                                max_n_connections=1000,
                                fixed_conn_per_l1_neuron=None,
                                sampling_method='lognormal',
                                eject_outliers=True,
                                aggregation='sum',
                                majority_vote=True)
    else:
        result = eval(clf_name + "()")

    end = time.perf_counter()
    init_time = end-start
    return result, init_time


def call_fit(clf, name, data):

    if name == 'DBSCAN':

        start = time.perf_counter()
        clustering = clf.fit(data)
        labels_temp = np.where(clustering.labels_ == -1, 1, 0)
        end = time.perf_counter()
        train_time = end-start

        return labels_temp, train_time

    else:
        s1 = time.perf_counter()
        clf = clf.fit(data)
        e1 = time.perf_counter()
        train_time = e1-s1

        return clf, train_time


def call_predict(clf, clf_name, data):
    start = time.perf_counter()

    if clf_name == 'IsolationForest':
        labels = np.where(clf.predict(data) == -1, 1, 0)
    else:
        labels = clf.predict(data)

    if clf_name == 'Perception':
        scores = clf.scores_
    elif clf_name == 'NeuralNetwork':
        scores = clf.scores_
    elif clf_name == 'IsolationForest':
        scores = clf.decision_function(data)*-1
    else:
        scores = clf.decision_function(data)

    end = time.perf_counter()
    predict_time = end-start

    return labels, scores, predict_time


def apply_classifiers(classifiers, dataset_name,
                      predict_data, predict_labels,
                      train_data=None):

    metrics_df = None
    labels = np.empty(len(predict_data))
    scores = np.array(len(predict_data))

    # run the classifiers over the data and produce results
    for clf_name in classifiers:

        print('classifier in progress: {}'.format(clf_name))

        if ((dataset_name == "http") and (clf_name == "OCSVM")):
            continue  # takes too long to run

        elif (dataset_name == 'credit-card') and\
                (clf_name == "LOF" or
                 clf_name == "KNN" or
                 clf_name == "OCSVM"):
            continue  # takes too long to run

        elif clf_name == 'DBSCAN':

            # initialise classifier
            clf, init_time = call_init_classifier(clf_name)

            # fit classifier (predict not required)
            labels, train_time = call_fit(clf, clf_name, predict_data)
            scores = None
            predict_time = 0

        else:
            # initialise classifier
            clf, init_time = call_init_classifier(clf_name, train_data)

            # fit classifier
            clf, train_time = call_fit(clf, clf_name, train_data)

            # predict using classifier to get labels and scores
            labels, scores, predict_time = call_predict(
                clf, clf_name, predict_data)

        # all these three need to be set
        total_time = init_time + train_time + predict_time

        print("total run time: {}".format(total_time))

        if clf_name == 'NeuralNetwork':
            print("data size: {}".format(len(train_data)))
            print("total data samples used: {}".format(
                clf.l0_l1_sum_connections_))
            print("% of data used: {}".format(
                clf.l0_l1_sum_connections_/len(train_data)*100))

        # produce the reports
        # -------------------
        cls_report_dic = classification_report(
            predict_labels, labels,
            target_names=['normal', 'abnormal'], output_dict=True)

        if clf_name == 'DBSCAN':
            # show binary label version of auc score
            auc_score = roc_auc_score(predict_labels, labels)
        else:
            auc_score = roc_auc_score(predict_labels, scores)

        # collect all the metrics
        metrics_temp = pd.DataFrame({
            'Dataset': [dataset_name],
            'Classifier': [clf_name],
            'Precision': [cls_report_dic['abnormal']['precision']],
            'Recall': [cls_report_dic['abnormal']['recall']],
            'F1': [cls_report_dic['abnormal']['f1-score']],
            'Support': [cls_report_dic['abnormal']['support']],
            'AUC': [auc_score],
            'Init. time': [init_time],
            'Train time': [train_time],
            'Predict time': [predict_time],
            'Runtime': [total_time],
        })

        metrics_df = pd.concat([metrics_df, metrics_temp]
                               ).reset_index(drop=True)

    # remove the first row due to timing issues it is duplicated
    metrics_df = metrics_df.iloc[1:]

    return metrics_df


def format_df(metrics_df):
    # round required columns to decimal places, leave time as is
    list_round_2 = ['Precision', 'Recall', 'F1', 'AUC']
    list_round_5 = ['Init. time', 'Train time', 'Predict time', 'Runtime']
    metrics_df[list_round_2] = metrics_df[list_round_2].round(2)
    metrics_df[list_round_5] = metrics_df[list_round_5].round(5)

    # drop columns not required
    columns_to_drop = ['Dataset', 'Init. time',
                       'Train time', 'Predict time', 'Support']
    metrics_df_dropped = metrics_df.drop(columns_to_drop, axis=1)
    df = metrics_df_dropped.copy()

    return df

# highlight bold for latex table use


def highlight_max_min(df_, col_show_max):

    df = df_.copy()
    # produce print out for paper
    for col, v in col_show_max.items():

        if v:
            # highlight maximum of columns
            maximum = df[col].max()
            bolded = df[col].apply(lambda x: "\\textbf{%s}" % x)
            df[col] = df[col].where(df[col] != maximum, bolded)
        else:
            # highlight minimum of columns
            minimum = df[col].min()
            bolded = df[col].apply(lambda x: "\\textbf{%s}" % x)
            df[col] = df[col].where(df[col] != minimum, bolded)

    return df


def highlight_rows_max_min(df_, col_agg):

    df = df_.copy()

    df_temp = df.drop(['Dataset'], axis=1)

    for index, row in df_temp.iterrows():

        if col_agg is True:
            agg = max(row)
        else:
            agg = min(row)

        df.loc[index] = df.loc[index].where(
            df.loc[index] != agg, "\\textbf{%s}" % agg)

    return df

# %%


# for row in df_f1.index:
#     print(f'Max element of row {row} is:', max(df_f1.iloc[row]))
#     break

# %%


# bold = df_f1.apply(lambda x: "\\textbf{%s}" % x, axis=1)

# max_in_rows = df_f1.max(axis=1)

# df = df_f1.drop(['Dataset'], axis=1)

# for index, row in df.iterrows():
#     # print(df.iloc[row])
#     maximum = max(row)
#     print(maximum)

#     # df_f2.loc[row] = df.iloc[row].where(df.iloc[row] != maximum, "\\textbf{%s}")
#     df_f1.loc[index] = df_f1.loc[index].where(
#         df_f1.loc[index] != maximum, "\\textbf{%s}" % maximum)


# %%
# for index, row in df_f1.iterrows():
#     bolded = df_f1.apply(lambda x: "\\textbf{%s}" % x)

#     df_f1.iloc[[index]] = df_f1.iloc[[index]].where(
#         df_f1.iloc[[index]] != 0.618, bolded)

#     print(df_f1.iloc[[index]].max(axis=1))
#     break
