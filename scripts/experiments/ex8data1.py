#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 17 2021.

@author: Nassir, Mohammad
"""

# %% imports
from perception import Perception
from neural_network import Neural_network
from nn_utilities.reporting_and_plotting import produce_report
from nn_utilities.reporting_and_plotting import report_acc_over_neurons
from nn_utilities.reporting_and_plotting import report_acc_over_subsamples
from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat

from pyod.models.auto_encoder import AutoEncoder

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

# %% ex8data1
# Andrew Ng. Machine learning: Programming Exercise 8: Anomaly Detection and
# Recommender Systems. 2012.

data_path = "../../data/ex8data1.mat"
data = loadmat(data_path)
X = data['X']
X_val = data['Xval']
y_val = data['yval']

# show the headers
for key, val in data.items():
    print(key)

print("The shape of X is: {}".format(X.shape))
print("The shape of X_val is: {}".format(X_val.shape))
print("The shape of y is: {}".format(y_val.sum()))

# %% Standarise the validation data only (training data not used)
sc = StandardScaler(with_mean=True)
sc.fit(X_val)
X_val = sc.transform(X_val)

# %% plot the data with labels
fig, ax = plt.subplots(figsize=(12, 8))

X_val_anomalies = X_val[np.where(y_val == 1)[0]]
X_val_normal = X_val[np.where(y_val == 0)[0]]

ax = sns.scatterplot(x=X_val_anomalies[:, 0],
                     y=X_val_anomalies[:, 1], color='darkorange',
                     label='Anomaly', marker=',', zorder=1)
ax = sns.scatterplot(x=X_val_normal[:, 0], y=X_val_normal[:, 1],
                     alpha=0.3, label='Normal', zorder=0)
plt.legend()

# %% snn model anomaly detection
clf = Perception()
clf.fit_predict(X_val)
produce_report(clf.labels_, y_val, X_val, clf.scores_,
               save_path=image_save_path + 'ex8data1_validation.png')

# %% plot the scores
plt.scatter(range(len(clf.scores_)), clf.scores_)

# %% net model for anomaly detection

# the net is not expected to work as well as the data size is quite small.
# In such a case a single neuron model is likely more appropriate as the
# results also support.
net = Neural_network(n0=len(X_val),
                     n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=True,
                     aggregation='sum',
                     majority_vote=True)
net.fit(X_val)
net.predict(X_val)
produce_report(net.labels_, y_val, X_val, net.scores_)

# %% auc score for scores - net
fpr, tpr, thresholds = roc_curve(y_val, net.scores_, pos_label=1)
plt.plot(fpr, tpr)

auc_score = roc_auc_score(y_val, net.scores_)
print('auc score for net labels: {}'.format(auc_score))

# %% fixed net model for anomaly detection
net = Neural_network(n0=len(X_val),
                     n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=False,
                     aggregation='sum',
                     majority_vote=True,
                     fixed_conn_per_l1_neuron=300)

net.fit(X_val)
net.predict(X_val)

produce_report(net.labels_, y_val, X_val, net.scores_)

# %% plot the scores
plt.scatter(range(len(net.anomaly_sum_l2_)), net.scores_)

# %% pyod isolation forest contamination ration is default 0.1
clf_if = IForest(contamination=0.1)
clf_if.fit(X_val)
scores = clf_if.decision_function(X_val)

produce_report(clf_if.labels_, y_val, X_val, scores,
               save_path=None)

# %% SKlearn isolation forest new in 0.22: contamination ratio set to 'auto'
clf_if2 = IsolationForest()
clf_if2.fit(X_val)
labels = np.where(clf_if2.predict(X_val) == -1, 1, 0)
scores = clf_if2.decision_function(X_val)*-1

produce_report(labels, y_val, X_val, scores,
               save_path=None)

# %% autoencoder
clf_a = AutoEncoder(hidden_neurons=[2, 64, 64, 2])
clf_a.fit(X_val)

scores = clf_a.decision_function(X_val)

produce_report(clf_a.labels_, y_val, X_val, scores,
               save_path=None)

# %% accuracy (AUC score) over n_neurons
report_acc_over_neurons(
    train_data=X_val, predict_data=X_val, true_labels=y_val,
    plot_title=None,  # "AUC score over number of neurons",
    x_axis_label="Number of neurons", y_axis_label="AUC score",
    n_runs=1, n_neurons=1000,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data1_n_neurons.png')

# %% accuracy (AUC score) over n_subsamples
report_acc_over_subsamples(
    train_data=X_val, predict_data=X_val, true_labels=y_val,
    plot_title=None,  # "AUC scores over subsample size",
    x_axis_label="Subsample size", y_axis_label="AUC score",
    n_runs=1, n_neurons=n_neurons, n_connections=300,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data1_subsample_size.png')
