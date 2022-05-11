# !/usr/bin/env python3.
# -*- coding: utf-8 -*-

"""Created on Dec 17 2021.

@author: Nassir, Mohammad.
"""

# %% setup
from perception import Perception
from neural_network import Neural_network
from utilities import plot_data
import numpy as np
import pandas as pd

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

# %% Read Galton heights
galton_height_path = "../../data/Galton.txt"
g_df = pd.read_csv(galton_height_path, sep='\t', lineterminator='\r')
data = g_df['Height'].values

# replicate data if required
for _ in range(0):
    data = np.concatenate([data, data])
    # data = np.concatenate([data, galton_heights])
    # ldata = np.concatenate([data, np.array([600, 700, 750, 800, 980])])

galton_save_path = image_save_path + 'galton_heights.png'
plot_data(data, galton_save_path, save_switch=image_save_switch,
          balls_height=2.8,
          x_label='Height', y_label='Frequency')

# %% Single neuron model
clf = Perception()
clf.fit_predict(data)
print(np.unique(clf.anomalies_))

# %% net anomaly detection
net = Neural_network(n0=len(data), n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=True,
                     aggregation='sum',
                     majority_vote=True)

net.fit(data)
net.predict(data)
print(np.unique(np.sort(net.output_anomalies_)))

# %% append an extreme anomaly to the original data
append_data = np.array([700])
e_data = np.append(data, append_data)

plot_data(e_data, balls_height=10, x_label='Height', y_label='Frequency')

# %% snn model on data with one extreme outlier
clf = Perception()
clf.fit_predict(e_data)
print(np.unique(clf.anomalies_))

# %% net model on data with one extreme outlier
net = Neural_network(n0=len(e_data), n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=True,
                     aggregation='sum',
                     majority_vote=True)
net.fit(e_data)
net.predict(e_data)
print(np.unique(np.sort(net.output_anomalies_)))

# %% append another extreme anomaly to the original data
append_data = np.array([200, 250])  # append_data = np.array([200, 25000])
e_data2 = np.append(data, append_data)
plot_data(e_data2, balls_height=5, x_label='Height', y_label='Frequency')

# %% snn model extreme anomaly detection
clf = Perception()
clf.fit_predict(e_data2)
print(np.unique(np.sort(clf.anomalies_)))

# %% net model extreme anomaly detection
net = Neural_network(n0=len(e_data2), n1=n_neurons,
                     min_n_connections=min_c,
                     max_n_connections=max_c,
                     sampling_method='lognormal',
                     eject_outliers=True,
                     aggregation='sum',
                     majority_vote=True)
net.fit(e_data2)
net.predict(e_data2)
print(np.unique(np.sort(net.output_anomalies_)))
