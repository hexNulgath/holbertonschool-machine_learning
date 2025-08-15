#!/usr/bin/env python3
"""gp"""
import numpy as np


class GaussianProcess:
    """
    represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1)
        representing the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1)
        representing the outputs of the black-box function for each
        input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output
        of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        """
        X1 = X1.flatten()
        X2 = X2.flatten()
        rbf = np.exp(-0.5 * (np.subtract.outer(X1, X2) ** 2) / self.l ** 2)
        return self.sigma_f ** 2 * rbf
