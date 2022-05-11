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
from sklearn.metrics import roc_curve
from pyod.models.iforest import IForest

from nn_utilities.reporting_and_plotting import report_acc_over_neurons
from nn_utilities.reporting_and_plotting import report_acc_over_subsamples
from perception import Perception
from neural_network import Neural_network

from scipy.io import loadmat
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
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

print(X2.shape, X2_val.shape, y2_val.shape)

# show the headers
for key, val in data.items():
    print(key)

print("The shape of X is: {}".format(X2.shape))
print("The shape of X_val is: {}".format(X2_val.shape))
print("The shape of y is: {}".format(y2_val.sum()))

# %% get the mean and standard deviation and plot the data
multi_d_median = np.median(X2, axis=0)
# multi_d_median = np.expand_dims(multi_d_median, axis=0)
sd = np.std(X2, axis=0)

print(multi_d_median)
print(sd)

print(X2.shape)
print(multi_d_median.shape)

# plot difference between median and data points
data = cdist(X2,
             np.expand_dims(multi_d_median, axis=0),
             'euclidean'
             ).ravel()

data2 = cdist(X2_val,
              np.expand_dims(multi_d_median, axis=0),
              'euclidean'
              ).ravel()

# seaborn histogram
f, axes = plt.subplots(figsize=(12, 6))
sns.histplot(data, kde=False, bins=10, color='grey',
             alpha=0.2, edgecolor='black')
sns.scatterplot(x=data, y=np.zeros_like(data)+2.8, color='black', marker='o',
                edgecolor='black', alpha=1, s=40)

colors = np.where(y2_val == 1)
sns.scatterplot(x=data2, y=np.zeros_like(data2)+50, marker='o',
                edgecolor='white', alpha=1, s=40, c=y2_val)

# Add labels
plt.title('ex8data2')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

# %% make and add the extreme anomalies
e_anomalies = np.array([])

number_of_std = 5
for i in range(5):

    # make 11 dimensional random number
    random_element = np.random.uniform(0, 1, 11)

    if i % 2 == 0:
        temp_anom = multi_d_median + number_of_std*sd + random_element
    else:
        temp_anom = multi_d_median - number_of_std*sd - random_element

    # print(temp_anom)
    # print(temp_anom.shape)
    # temp_anom = np.reshape(temp_anom, (-1,len(temp_anom)))
    # print(temp_anom.shape)

    # add the anomaly
    e_anomalies = np.concatenate([e_anomalies, temp_anom])

    # j = min(j+1,10)

e_anomalies = np.reshape(e_anomalies, (-1, 11))
print(e_anomalies)

print("Original X2 shape: {}".format(X2.shape))

X2 = np.append(X2, e_anomalies, axis=0)

print("New X2 shape: {}".format(X2.shape))

# %% scale the data after adding the new anomalies
sc = StandardScaler()
sc.fit(X2)
X2 = sc.transform(X2)
X2_val = sc.transform(X2_val)

# %% snn model
clf = Perception()
clf.fit(X2)
clf.predict(X2_val)

print(classification_report(
    y2_val, clf.labels_,
    target_names=['normal', 'abnormal'], output_dict=False))

auc_score = roc_auc_score(y2_val, clf.scores_)
print('auc score: {}'.format(auc_score))

# %% net model for anomaly detection
net = Neural_network(n0=len(X2), n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     fixed_conn_per_l1_neuron=None,
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
    plot_title="AUC scores over number of neurons - extreme",
    y_axis_label="AUC score", x_axis_label="Number of neurons",
    n_runs=1, n_neurons=500,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data2_n_neurons_extreme.png')

# %% AUC scores over number of neurons - extreme
report_acc_over_subsamples(
    train_data=X2, predict_data=X2_val, true_labels=y2_val,
    plot_title="AUC scores over subsample size - extreme",
    y_axis_label="AUC score", x_axis_label="Subsample size",
    n_runs=1, n_neurons=n_neurons, n_connections=250,
    min_c=min_c, max_c=max_c,
    save_path=image_save_path + 'ex8data2_subsample_size_extreme.png')
