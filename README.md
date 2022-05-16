Neural Network for Anomaly Detection

This project contains the prototypical implemenation of the neural network described in the paper "A Vision Inspired Neural Network for Unsupervised Anomaly Detection in Unordered Data" 
that can be found at: http://arxiv.org/abs/2205.06716

The neural network detects global anomalies in univariate and multivariate data practically parameter-free! Just give it the standardised data as a numpy array and it will return
the label (normal or anomalous) of each example. The network also gives each example a score that is more useful for ranking anomalies, and for further analysis and investigation.

Presently, the network scores can be used to rank the data points. This works very well according to AUC metric on a range of publicly available data sets. However, the decisions 
output by the network requires improvement.

Notes on the network:

  Returns the anomaly label and associated score of each observation.

  The greater the score, above zero, the more anomalous an observation is.

  This function specialises to one-dimensional data when the data only has one feature.

In the multi-dimensional case:

   We take in standardised data and calculate each observations' distance from the median using a
   distance measure, for example Mahalanobis or Euclidean. Then, given the one-dimensional data, we apply the
   one-dimensional anomaly detection algorithm as before.
