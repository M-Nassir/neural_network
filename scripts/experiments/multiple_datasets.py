#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Dec 17 2021.

@author: Nassir, Mohammad
"""

# %% setup
from nn_utilities.classifying import get_file_data, apply_classifiers
from nn_utilities.classifying import highlight_rows_max_min
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import StandardScaler

# %% setup
image_save_switch = True

# %%
with open('../../paths.txt') as f:
    image_save_path = f.readline()[:-1]
    latex_table_save_path = f.readline()[:-1]

    print(image_save_path)
    print(latex_table_save_path)

# %% set parameters
min_c = 10
max_c = 1000
n_neurons = 256

# %% get properties of each data file adn write df to latex
base_path = "../../data/ODDS_multivariate/"
data_properties_df = None

# loop over datasets in directory
for file_name in os.listdir(base_path):

    dataset_name, X_original, y = get_file_data(base_path, file_name)

    # if file name is not supported, just continue
    if dataset_name is None:
        continue

    # write dataset summary to dataframe
    data_properties_temp = pd.DataFrame({
        'Name': [dataset_name],
        '\\# examples': [X_original.shape[0]],
        '\\# features': [X_original.shape[1]],
        '\\% anomalies': [float(np.round(y.sum()/X_original.shape[0]*100, 2))],
    })

    data_properties_df = pd.concat(
        [data_properties_df, data_properties_temp]).reset_index(drop=True)

# write the latex formatted table to paper location
if image_save_switch is True:
    data_properties_df.to_latex(
        latex_table_save_path + 'dataset_properties.tex',
        escape=False,
        index=False)

# %% apply classifiers to each dataset
classifiers = [
    'HBOS',  # to be ignored, first run in loop slower
    'HBOS',
    # 'IForest',
    'IsolationForest',
    # autoencoder TODO
    # 'KNN',
    # 'LOF',
    # 'MCD',
    # 'OCSVM',
    'Perception',
    'NeuralNetwork',
    # 'DBSCAN',
]

metrics_df = None

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    # loop over datasets in directory
    for file_name in os.listdir(base_path):

        dataset_name, X_original, y = get_file_data(base_path, file_name)

        if dataset_name is None:
            continue

        # scaling (very important to do)
        # scale to unit standard deviation along each feature + subtract mean?
        sc = StandardScaler(with_mean=True)  # False
        sc.fit(X_original)
        X = sc.transform(X_original)

        # Apply each classifier to dataset
        print("============================================")
        print('current file in progress ...: {}'.format(dataset_name))

        metrics_temp = apply_classifiers(classifiers, dataset_name,
                                         predict_data=X,
                                         predict_labels=y,
                                         train_data=X)

        metrics_df =\
            pd.concat([metrics_df, metrics_temp])

    metrics_df.reset_index(drop=True)

# %% pivot tables column wise (index as Dataset)


def get_metrics_col_save_to_latex(df_input, pivot_col, save_path,
                                  round_val, col_max=True):

    df_t = df_input[['Dataset', 'Classifier', pivot_col]]
    df_t = pd.pivot_table(df_t,
                          values=pivot_col,
                          index=['Dataset'],
                          columns='Classifier').reset_index()

    df_t.columns.name = None
    df_t = df_t.round(round_val)

    # save to latex and highlight max
    highlighted_df = highlight_rows_max_min(df_t, col_max)

    print(highlighted_df.to_latex(escape=False, index=False))

    # write the latex formatted table to paper location
    highlighted_df.to_latex(
        save_path,
        escape=False,
        index=False)

    return df_t


df_f1 = get_metrics_col_save_to_latex(
    metrics_df, pivot_col='F1',
    save_path=latex_table_save_path + 'f1_results_col.tex',
    round_val=3, col_max=True)

df_auc = get_metrics_col_save_to_latex(
    metrics_df, pivot_col='AUC',
    save_path=latex_table_save_path + 'auc_results_col.tex',
    round_val=2, col_max=True)

df_runtime = get_metrics_col_save_to_latex(
    metrics_df, pivot_col='Runtime',
    save_path=latex_table_save_path + 'runtime_results_col.tex',
    round_val=3, col_max=False)

# %%

df_precision = get_metrics_col_save_to_latex(
    metrics_df, pivot_col='Precision',
    save_path=latex_table_save_path + 'precision_results_col.tex',
    round_val=3, col_max=True)


df_recall = get_metrics_col_save_to_latex(
    metrics_df, pivot_col='Recall',
    save_path=latex_table_save_path + 'recall_results_col.tex',
    round_val=2, col_max=True)

# %% get metric results and save to latex (index as Classifier)


# def get_metrics_save_to_latex(df_input, pivot_col, save_path,
#                               round_val, col_max=True):

#     df_t = df_input[['Dataset', 'Classifier', pivot_col]]
#     df_t = pd.pivot_table(df_t,
#                           values=pivot_col,
#                           index=['Classifier'],
#                           columns='Dataset').reset_index()

#     df_t.columns.name = None
#     df_t = df_t.round(round_val)

#     # save to latex and highlight max
#     column_subset = df_t.columns.values.tolist()
#     column_subset.remove('Classifier')

#     if col_max:
#         col_show_max = {i: True for i in column_subset}
#     else:
#         col_show_max = {i: False for i in column_subset}

#     highlighted_df = highlight_max_min(df_t, col_show_max)

#     print(highlighted_df.to_latex(escape=False, index=False))

#     # write the latex formatted table to paper location
#     highlighted_df.to_latex(
#         save_path,
#         escape=False,
#         index=False)

#     return df_t


# df_precision = get_metrics_save_to_latex(
#     metrics_df, pivot_col='Precision',
#     save_path=latex_table_save_path + 'precision_results.tex',
#     round_val=3, col_max=True)


# df_recall = get_metrics_save_to_latex(
#     metrics_df, pivot_col='Recall',
#     save_path=latex_table_save_path + 'recall_results.tex',
#     round_val=2, col_max=True)

# df_f1 = get_metrics_save_to_latex(
#     metrics_df, pivot_col='F1',
#     save_path=latex_table_save_path + 'f1_results.tex',
#     round_val=3, col_max=True)

# df_auc = get_metrics_save_to_latex(
#     metrics_df, pivot_col='AUC',
#     save_path=latex_table_save_path + 'auc_results.tex',
#     round_val=2, col_max=True)

# df_runtime = get_metrics_save_to_latex(
#     metrics_df, pivot_col='Runtime',
#     save_path=latex_table_save_path + 'runtime_results.tex',
#     round_val=3, col_max=False)
