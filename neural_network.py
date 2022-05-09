"""Neural Network Anomaly Detection."""
# -*- coding: utf-8 -*-
# Author: Nassir, Mohammad <nassir.mohammad@airbus.com>
# License: BSD 2 clause

from perception import Perception
import random
import numpy as np


class Neural_network:
    """Neural network for anomaly detection.

    Return the anomaly scores and label of each one-dimensional or
    multi-dimensional example.

    ***

    Input data must be standardised if it is multidimensional,
    using for example: sklearn.preprocessing.StandardScaler.

    ***


    Assumptions on data distribution
    --------------------------------
    The neural network detects anomalies in one or more dimensions under
    the assumption that:

        'normal' data is found within one majority grouping/cluster.

        the anomalies are both rare and relatively far from the main group.

        anomalies can be discovered using the available features.

        the feature values are numerical.

        the input is a numpy array: rows correspond to each example, and
        columns correspond to each feature.

        the data has already been normalised to zero mean and standard
        deviation of 1 independently for all features.

        each neuron makes prediction on all examples.

        each neuron does random variable subsampling over a log-normal
        distribution.

        each neuron produces a score and binary output decision.

        all neural computations are summed by output nodes.

        decisions of each output neuron is > 0 for anomaly, otherwise normal.


    Parameters
    ----------
    n0 : integer
        The number of input receptor nodes. Equivalent to the length of data.

    n1 : integer, optional (default=256)
        The number of neurons in layer 1 of the network.

    min_n_connections : integer optional (default=10)
        The minimum number of connections a neuron in layer 1 has with
        input receptor nodes in layer 0, i.e., minimum number of subsamples
        to take from the input data.

    max_n_connections : integer optional (default=1000)
        The maximum number of connections a neuron in layer 1 has with
        input receptor nodes in layer 0, i.e., maximum number of subsamples
        to take from the input data.

    fixed_conn_per_l1_neuron : integer, optional (default=None)
        If this number is specified then every neuron in layer 1 connects
        to the same number of receptor nodes, i.e., each neuron takes the same
        number of random subsamples from the input data.

    sampling_method : string (default='lognormal') (optional: 'uniform')
        The distribution to use for sampling the input data.

    eject_outliers : Bool (default=True)
        Whether or not a neuron ejects outliers from its subsample before
        relearning its parameters.

    aggregation : string (default='sum') (optional: 'max', 'median')
        Method for aggregating neuron scores at the output node.

    majority_vote = Bool (default=True)
        Whether to use majority vote of neuron decisions at the output nodes
        to decide if an input data example is anomalous.


    Attributes
    ----------
    c_min_ : integer
        The minimum number of active connections a neuron in layer 1 has with
        input receptor nodes in layer 0, i.e., minimum number of subsamples
        to take from the input data. This is the smaller of min_n_connections
        and length of data.

    c_max_ : integer
        The maximum number of active connections a neuron in layer 1 has with
        input receptor nodes in layer 0, i.e., maximum number of subsamples
        to take from the input data. This is the smaller of max_n_connections
        and length of data.

    subsample_integers_ : numpy array of integers of length n1
        The subsample integers for each neuron in l1, i.e., the number of
        subsamples each neuron takes of the input data.

    l0_l1_sum_connections_ : integer
        The total number of active connections between receptors and neurons.
        Equal to np.sum(subsample_integers_)

    random_index_subsamples_ : list of lists
        Holds the randomly selected index positions of input data to be
        subsampled by all the neurons.

        Each member of the list holds the subsample of input data to be input
        to a particular neuron. The subsample is taken by data index position.

    classifiers_ : list
        List to hold fitted perception classifier information by each neuron.

    anomaly_sum_l2_ : numpy array
        output node summation of anomaly scores (computed by each neuron)
        for each example.

    labels_ : numpy array of size len(data)
        The network decisions on whether a sample is anomalous or normal.

    output_anomalies_ : numpy array
        The original input data values decided by the network to be anomalous.

    scores_ : numpy array
        output node summation of anomaly scores (computed by each neuron)
        for each example. (Same as anomaly_sum_l2_)
    """

    def __init__(self, n0,
                 n1=256,
                 min_n_connections=10,
                 max_n_connections=1000,
                 fixed_conn_per_l1_neuron=None,
                 sampling_method='lognormal',
                 eject_outliers=True,
                 aggregation='sum',
                 majority_vote=True):

        # checks
        # --------

        # assert n0 (length_of_data) is positive integer
        assert (n0 > 0) and (type(n0) is int)

        # assert n1 is positive integer
        assert (n1 > 0) and (type(n1) is int)

        # assert min_n_connections is positive integer
        assert (min_n_connections > 0) and (type(min_n_connections) is int)

        # assert max_n_connections is positive integer
        assert (max_n_connections > 0) and (type(max_n_connections) is int)

        # assert min_n_connections < max_n_connections
        # (otherwise we have sampling problems)
        assert min_n_connections < max_n_connections

        if fixed_conn_per_l1_neuron is not None:
            assert fixed_conn_per_l1_neuron <= n0, \
                "number of fixed subsamples is greater than data size"

        # assign net parameters
        # -------------------------

        # number of input nodes/output nodes - layer 0 and layer 2
        self.n0 = n0

        # number of neurons - layer 1
        self.n1 = n1

        # assign minimum/maximum number of connections
        self.c_min_ = min(min_n_connections, self.n0)
        self.c_max_ = min(max_n_connections, self.n0)

        if self.c_min_ == self.c_max_:
            assert 1 == 0, \
                "Data size is too small to specifiy sub-sample size parameters"

        self.fixed_conn_per_l1_neuron = fixed_conn_per_l1_neuron
        self.sampling_method = sampling_method
        self.eject_outliers = eject_outliers
        self.aggregation = aggregation
        self.majority_vote = majority_vote

    def fit(self, data):
        """Fit the network model to the data.

        Parameters
        ----------
        data : numpy array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        # do checking
        # --------------

        assert len(data) == self.n0, \
            "data length does not match init() parameter self.n0"

        # subsampling
        # --------------

        if self.fixed_conn_per_l1_neuron is not None:

            # do fixed connection model
            # -------------------------
            self.subsample_integers_ = np.full(
                self.n1, self.fixed_conn_per_l1_neuron)
        else:

            # do random subsample connection model
            # ------------------------------------
            if self.sampling_method == 'lognormal':

                # initialise parameters for log-normal distribution
                mu, sigma = 3, 2
                subsample_list = []
                i = 0
                while i < self.n1:

                    # take self.n1 (100) subsamples and shift by self.c_min
                    snt = np.random.lognormal(mu, sigma, self.n1) + self.c_min_

                    # select n1 subsample sizes between c_min and c_max
                    snt = snt[(snt >= self.c_min_) & (snt <= self.c_max_)]

                    # add lognormal numbers array to list if non-empty array
                    # do n1 times, it produces more integers than required,
                    # but is simpler than counting total number of elements.
                    if snt.size != 0:
                        i = i + snt.size
                        subsample_list.append(snt)

            elif self.sampling_method == 'uniform':
                subsample_list = []
                snt = np.random.uniform(
                    self.c_min_, self.c_max_, self.n1)
                subsample_list.append(snt)
            else:
                assert False, "sampling method not supported"

            # flatten the list, convert to integers and take only n1 samples
            self.subsample_integers_ = \
                np.concatenate(subsample_list).astype(int)[:self.n1]

        # check number of subsamples matches the number of neurons
        assert len(self.subsample_integers_) == self.n1

        # the total number of active connections between receptors and neurons
        self.l0_l1_sum_connections_ = np.sum(self.subsample_integers_)

        # subsample data to be taken by each neuron from input data
        self.random_index_subsamples_ = []
        for el in self.subsample_integers_:
            temp = [random.randint(0, self.n0-1) for _ in range(el)]
            self.random_index_subsamples_.append(temp)

        # fit each neuron model
        # ---------------------

        self.classifiers_ = []

        # for each neuron
        for i in range(self.n1):

            # get the subsampled data for each neuron
            l1_neuron_sample = data[self.random_index_subsamples_[i]]

            # fit each neuron model

            # learn the parameters S, W, training
            # _median_, rounding_multiplier_,
            # (note median is the one after rounding multiplier)
            # e.g. [2.1, 2.3, 2.4] -> S = 68, W=3, med=23, multiplier=10,
            # [5,6,7]-> S = 18, W=3, med=6, multiplier=1

            clf = Perception()
            clf.fit(l1_neuron_sample)

            if self.eject_outliers:

                temp_res = clf.predict(l1_neuron_sample)
                mask = np.ones(len(l1_neuron_sample), dtype=bool)
                mask[np.where(temp_res == 1)] = False
                result = l1_neuron_sample[mask]

                clf = Perception()
                clf.fit(result)

            self.classifiers_.append(clf)

        return self

    def predict(self, data, new_data_flag=False):
        """Predict anomaly decision and scores for each input data example.

        Parameters
        ----------
        data : numpy array
            The data to make predictions upon.

        Returns
        -------
        labels_ : binary numpy array
            The labels of each input example: 0 normal, 1 anomaly.

        """
        temp_anomaly_scores = []

        for i in range(self.n1):

            # get the neuron model
            clf = self.classifiers_[i]

            # l1 neuron predict on all input data
            clf.predict(data)

            # get each neuron prediction scores and append to the array
            temp_anomaly_scores.append(clf.scores_)

        self.anomaly_scores_l1 = np.concatenate(temp_anomaly_scores)

        self.anomaly_scores_l1 = np.reshape(
            self.anomaly_scores_l1, (-1, len(data)))

        assert self.anomaly_scores_l1.shape == (self.n1, len(data))

        # -------------------------------------
        # compute scores by aggregation method
        # -------------------------------------

        if self.aggregation == 'max':  # does not produce good results. Remove.
            self.anomaly_sum_l2_ = self.anomaly_scores_l1.max(axis=0)
        elif self.aggregation == 'median':
            self.anomaly_sum_l2_ = np.median(self.anomaly_scores_l1, axis=0)
        elif self.aggregation == 'sum':
            # division required when comparing over different number of
            # neurons in different networks
            self.anomaly_sum_l2_ = self.anomaly_scores_l1.sum(axis=0)/self.n1
        else:
            # default is sum and divide by number of neurons
            self.anomaly_sum_l2_ = self.anomaly_scores_l1.sum(axis=0)/self.n1

        # ------------------------------
        # compute labels
        # ------------------------------

        if self.majority_vote is True:
            # compute labels using sum of binary labels
            self.anomaly_labels_l1 =\
                np.where(self.anomaly_scores_l1 > 0, 1, -1)

            assert self.anomaly_labels_l1.shape == (self.n1, len(data))

            votes = self.anomaly_labels_l1.sum(axis=0)

            self.labels_ = np.where(votes >= 0, 1, 0)

        else:
            # compute labels using sum of scores only
            self.labels_ = np.where(self.anomaly_sum_l2_ > 0, 1, 0)

        # re-assign the predicted anomalies
        self.output_anomalies_ = data[np.where(self.labels_ == 1)]

        # re-assign layer 2 prediction scores for documentation consistency
        self.scores_ = self.anomaly_sum_l2_

        return self.labels_
