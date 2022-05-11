#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 17 2021.

@author: Nassir, Mohammad
"""

# TODO: contour plots

# %% imports
from perception import Perception
from neural_network import Neural_network
from nn_utilities.reporting_and_plotting import produce_report
from nn_utilities.reporting_and_plotting import report_acc_over_neurons
from nn_utilities.reporting_and_plotting import report_acc_over_subsamples
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
from matplotlib import colors

# %% setup
image_save_switch = False

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

data_path = r"../../data/ex8data1.mat"
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

# %% add the extreme anomalies
X_val = np.append(X_val, np.array([[55, 72]]), axis=0)
y_val = np.append(y_val, np.array([[1]]), axis=0)

X_val = np.append(X_val, np.array(
    [[35, 80], [100, 150], [67, 74], [87, 75]]), axis=0)
y_val = np.append(y_val, np.array([[1], [1], [1], [1]]), axis=0)

X_val = np.append(X_val, np.array(
    [[102, 149], [101, 151], [102, 135], [106, 155]]), axis=0)
y_val = np.append(y_val, np.array([[1], [1], [1], [1]]), axis=0)

# %% standardise
sc = StandardScaler()
sc.fit(X_val)
X_val = sc.transform(X_val)

# %% plot the data with labels and extreme values
fig, ax = plt.subplots(figsize=(12, 8))

X_val_anomalies = X_val[np.where(y_val == 1)[0]]
X_val_normal = X_val[np.where(y_val == 0)[0]]

ax = sns.scatterplot(x=X_val_anomalies[:, 0], y=X_val_anomalies[:, 1],
                     color='darkorange', label='Anomaly', marker='o', zorder=1)
ax = sns.scatterplot(
    x=X_val_normal[:, 0], y=X_val_normal[:, 1], alpha=0.3, label='Normal',
    zorder=0)
plt.legend()

# %% snn model anomaly detection
# run snn on only the validation portion of data; including fit and predict
clf = Perception()
clf.fit_predict(X_val)
produce_report(clf.labels_, y_val, X_val, clf.scores_,
               save_path=image_save_path + 'ex8data1_extreme.png')

# %% net model for anomaly detection
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

# %% net model with contours
# make dataframe with all points and labels
m_df = pd.DataFrame(X_val, columns=[['f1', 'f2']])
m_df['y_label'] = y_val

xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
grid_data = np.c_[xx.ravel(), yy.ravel()]

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
m_df['net_label'] = net.labels_

Z = net.predict(grid_data)  # , new_data_flag=True)
Z = Z.reshape(xx.shape)

Z_scores = net.scores_*-1
Z_scores = Z_scores.reshape(xx.shape)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# fill blue colormap from minimum anomaly score to threshold value
ax.contourf(xx, yy, Z_scores, cmap=plt.cm.Blues_r)

# draw red contour line where anomaly score is equal to threshold
ax.contour(xx, yy, Z_scores, levels=[0],
           linewidths=2, linestyles='dashed',
           colors='red')

# # # colour inlier area orange
ax.contourf(xx, yy, Z, levels=[0, 0.9], colors='orange')

# scatter all points and colour with labels: white inliers, black anomalies
ax.scatter(m_df['f1'], m_df['f2'], c=m_df['net_label'],
           cmap=colors.ListedColormap(['green', 'black']))

# %%

# make dataframe with all points and labels
m_df = pd.DataFrame(X_val, columns=[['f1', 'f2']])
m_df['y_label'] = y_val

xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
grid_data = np.c_[xx.ravel(), yy.ravel()]

clf = Perception()
clf.fit_predict(X_val)

m_df['net_label'] = clf.labels_

Z = clf.predict(grid_data)
Z = Z.reshape(xx.shape)

Z_scores = clf.scores_*-1
Z_scores = Z_scores.reshape(xx.shape)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# fill blue colormap from minimum anomaly score to threshold value
ax.contourf(xx, yy, Z_scores, cmap=plt.cm.Blues_r)

# draw red contour line where anomaly score is equal to threshold
ax.contour(xx, yy, Z, levels=[0],
           linewidths=2, linestyles='dashed',
           colors='red')

# # # colour inlier area orange
ax.contourf(xx, yy, Z, levels=[0, 0.9], colors='orange')

# scatter all points and colour with labels: white inliers, black anomalies
ax.scatter(m_df['f1'], m_df['f2'], c=m_df['net_label'],
           cmap=colors.ListedColormap(['white', 'black']))

# %% SKlearn isolation forest new in 0.22: contamination ratio set to 'auto'
# 'auto' sometimes fails compeltely e.g. for gaussian data, but other cases
# seems to perform better than 0.1, explore in a separate paper.
clf_if2 = IsolationForest()
clf_if2.fit(X_val)

m_df = pd.DataFrame(X_val, columns=[['f1', 'f2']])
m_df['y_label'] = y_val
m_df['if_label'] = np.where(clf_if2.predict(X_val) == -1, 1, 0)

labels = np.where(clf_if2.predict(X_val) == -1, 1, 0)
scores = clf_if2.decision_function(X_val)*-1

produce_report(labels, y_val, X_val, scores,
               save_path=None)

# auc_score = roc_auc_score(y_val, scores)
# print('auc score for labels: {}'.format(auc_score))

xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
grid_data = np.c_[xx.ravel(), yy.ravel()]

clf_if2.predict(grid_data)

Z = clf_if2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.scatter(X_val[:, 0], X_val[:, 1])

# fill blue colormap from minimum anomaly score to threshold value
ax.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# draw red contour line where anomaly score is equal to threshold
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

# colour in the normal region
ax.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

ax.scatter(m_df['f1'], m_df['f2'], c=m_df['if_label'], cmap='Greys')

# %% AUC scores over number of neurons - extreme
report_acc_over_neurons(
    train_data=X_val, predict_data=X_val, true_labels=y_val,
    plot_title="AUC scores over number of neurons - extreme",
    y_axis_label="AUC score", x_axis_label="Number of neurons",
    n_runs=1, n_neurons=n_neurons,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data1_extreme_n_neurons.png')

# %% AUC scores over number of neurons - extreme
report_acc_over_subsamples(
    train_data=X_val, predict_data=X_val, true_labels=y_val,
    plot_title="AUC scores over subsample size - extreme",
    y_axis_label="AUC score", x_axis_label="Subsample size",
    n_runs=1, n_neurons=n_neurons, n_connections=200,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data1_extreme_subsample_size.png')
