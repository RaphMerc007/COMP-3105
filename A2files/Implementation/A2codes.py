# COMP 3105 Fall 2025 Assignment 2
# Raphael Mercier & Patrick Wu

from cvxopt import matrix, solvers
from scipy.optimize import minimize
import numpy as np  
import pandas as pd
import os 

#Q1 A
def minExpLinear(X, y, lamb):

    # Get # of elements and "features"
    n = X.shape[0]
    d = X.shape[1]

    # Define the loss function for the regularized ExpLinear loss
    def loss(params):

        # extract the first d elements from the params representing weights, and remaining element as the intercept
        w = params[:d]
        w0 = params[-1]

        # Get the margin of the i-th data point  
        margins = y * (X @ w + w0)

        # calculate and return loss
        term1 = np.maximum(0, -margins)
        term2 = np.exp(np.minimum(0, -margins))
        regularization = (lamb/2) * np.sum(w ** 2)
        return np.sum(term1 + term2) + regularization
    
    # initialize a zero (d + 1) x 1 vector storing the initial guess values
    initParams = np.zeros(d + 1)

    result = minimize(loss, initParams)
    
    # extract results from the scipy minimize function
    w = result.x[:d]
    w0 = result.x[-1]

    # Return weights and scalar intercept corresponding to the solution
    return w, w0