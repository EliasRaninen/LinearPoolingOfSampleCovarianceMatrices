# Linear Pooling Of Sample Covariance Matrices

This repository contains MATLAB code for the paper E. Raninen, D. E. Tyler and
E. Ollila, "Linear pooling of sample covariance matrices," in *IEEE Transactions
on Signal Processing*, Vol 70, pp. 659-672, 2021, doi:
[10.1109/TSP.2021.3139207](https://doi.org/10.1109/TSP.2021.3139207).

## Abstract

We consider the problem of estimating high-dimensional covariance matrices of
*K*-populations or classes in the setting where the sample sizes are comparable
to the data dimension. We propose estimating each class covariance matrix as a
distinct linear combination of all class sample covariance matrices. This
approach is shown to reduce the estimation error when the sample sizes are
limited, and the true class covariance matrices share a somewhat similar
structure. We develop an effective method for estimating the coefficients in the
linear combination that minimize the mean squared error under the general
assumption that the samples are drawn from (unspecified) elliptically symmetric
distributions possessing finite fourth-order moments. To this end, we utilize
the spatial sign covariance matrix, which we show (under rather general
conditions) to be an asymptotically unbiased estimator of the normalized
covariance matrix as the dimension grows to infinity. We also show how the
proposed method can be used in choosing the regularization parameters for
multiple target matrices in a single class covariance matrix estimation problem.
We assess the proposed method via numerical simulation studies including an
application in global minimum variance portfolio optimization using real stock
data.

## MATLAB scripts
* `linpool.m` : script for computing the covariance matrix estimators. Supports real and complex data.
* `example.m` : reproduces the mixed case from Table I of the paper.

## Authors
* Elias Raninen, Doctoral Candidate, Department of Signal Processing and Acoustics, Aalto University.
* David E. Tyler, Distinguished Professor, Department of Statistics, Rutgers University.
* Esa Ollila, Professor, Department of Signal Processing and Acoustics, Aalto University.
