#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Created on Dec 17 2021.

@author: Nassir Mohammad
"""

# %% imports
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pyod.models.iforest import IForest

from nn_utilities.reporting_and_plotting import report_acc_over_neurons
from nn_utilities.reporting_and_plotting import report_acc_over_subsamples
from perception import Perception
from neural_network import Neural_network

from scipy.io import loadmat
import numpy as np

# %% setup
image_save_switch = True

# %%
with open('../../paths.txt') as f:
    image_save_path = f.readline()
    image_save_path = image_save_path[:-1]
    print(image_save_path)

# %% set parameters
min_c = 10
max_c = 1000
n_neurons = 256

# %% read in the data
data_path = "../../data/ex8data2.mat"
data = loadmat(data_path)
X2 = data['X']
X2_val = data['Xval']
y2_val = data['yval']

sc = StandardScaler()
sc.fit(X2)
X2 = sc.transform(X2)
X2_val = sc.transform(X2_val)

print(X2.shape, X2_val.shape, y2_val.shape)

# show the headers
for key, val in data.items():
    print(key)

print("The shape of X is: {}".format(X2.shape))
print("The shape of X_val is: {}".format(X2_val.shape))
print("The shape of y is: {}".format(y2_val.sum()))

# %% snn model
clf = Perception()
clf.fit(X2)
clf.predict(X2_val)

validation_labels = clf.labels_
print(classification_report(
    y2_val, validation_labels,
    target_names=['normal', 'abnormal'], output_dict=False))

auc_score = roc_auc_score(y2_val, clf.scores_)
print('auc score: {}'.format(auc_score))

# %% net model for anomaly detection
net = Neural_network(n0=len(X2), n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=True,
                     aggregation='sum',
                     majority_vote=True)

net.fit(X2)
net.predict(X2_val)

print(classification_report(
    y2_val, net.labels_,
    target_names=['normal', 'abnormal'], output_dict=False))

auc_score = roc_auc_score(y2_val, net.scores_)
print('auc score: {}'.format(auc_score))

# %% pyod isolation forest contamination ration is default 0.1
clf_if = IForest()
clf_if.fit(X2)
scores = clf_if.decision_function(X2_val)
labels = clf_if.predict(X2_val)

# print the performance reports
print(classification_report(y2_val, labels,
                            target_names=['normal', 'abnormal'],
                            output_dict=False))

auc_score = roc_auc_score(y2_val, scores)
print('auc score for labels: {}'.format(auc_score))

# %% Sklearn IF
clf_if2 = IsolationForest()
clf_if2.fit(X2)
labels = np.where(clf_if2.predict(X2_val) == -1, 1, 0)
scores = clf_if2.decision_function(X2_val)*-1

# print the performance reports
print(classification_report(y2_val, labels,
                            target_names=['normal', 'abnormal'],
                            output_dict=False))

auc_score = roc_auc_score(y2_val, scores)
print('auc score for labels: {}'.format(auc_score))

# %% AUC scores over number of neurons - extreme
report_acc_over_neurons(
    train_data=X2, predict_data=X2_val, true_labels=y2_val,
    plot_title=None,  # "AUC scores over number of neurons",
    y_axis_label="AUC score", x_axis_label="Number of neurons",
    n_runs=1, n_neurons=1000,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data2_n_neurons.png')

# %% AUC scores over number of neurons - extreme
report_acc_over_subsamples(
    train_data=X2, predict_data=X2_val, true_labels=y2_val,
    plot_title=None,  # "AUC scores over subsample size",
    y_axis_label="AUC score", x_axis_label="Subsample size",
    n_runs=1, n_neurons=n_neurons, n_connections=250,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data2_subsample_size.png')
