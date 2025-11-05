# COMP 3105 Fall 2025 Assignment 3
# Raphael Mercier & Patrick Wu

from cvxopt import matrix, solvers
from scipy.optimize import minimize
from scipy.special import logsumexp
import numpy as np  
import pandas as pd
from A3helpers import convertToOneHot

#TODO: note: q1 a, b, c and correct for the testbed cases most of the time, maybe due to the model generation being random
# due to the removal of the random zeed by the prof. Double check correctness later tho

#Q1 a
def minMulDev(X, Y):
    n, d = X.shape
    # Get the k # of outputs per Y point
    k = Y.shape[1]

    # Objective function computing multinomial deviance loss
    def objective(params):
        #Flatten our matrix to a 1d array that can be used in the minimize function
        W = params.reshape(d, k)

        # Get the n x k matrix where each row contains k zHat values for each y
        zHat = X @ W

        loss = np.mean(logsumexp(zHat, axis = 1) - np.sum(Y * zHat, axis = 1))

        # Return average loss over all samples
        return loss

    # Initialize a (d x k) zero matrix storinig the initial guess
    initParams = np.zeros(d * k)

    result = minimize(objective, initParams)

    # Extract results and return a d x k matrix of optimal weights
    wStar = result.x.reshape(d, k)
    return wStar

#Q1 b
def classify(Xtest, W):

    # Get dimensions of inputs
    k = W.shape[1]

    # Get all the scores
    zHat = Xtest @ W

    # Find the class with the highest score for each sample
    predictedClasses = np.argmax(zHat, axis = 1)

    # Get one hot encoding for every sample, creating a m x k matrix
    yHat = convertToOneHot(predictedClasses, k)

    return yHat

#Q1 c
def calculateAcc(Yhat, Y):

    # Since Y's are one hot encoded, get max of k column indices for m row samples
    predictedClasses = np.argmax(Yhat, axis = 1)
    classes = np.argmax(Y, axis = 1)

    # Calculate accuracy, percentage of indices that match
    accuracy = np.mean(predictedClasses == classes)

    return accuracy