#!/usr/bin/env python3
"""11-gmm.py"""
import sklearn.mixture


def gmm(X, k):
    """
    calculates a GMM from a dataset
    """
    gmm_model = sklearn.mixture.GaussianMixture(
        n_components=k, covariance_type='full')
    gmm_model.fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
