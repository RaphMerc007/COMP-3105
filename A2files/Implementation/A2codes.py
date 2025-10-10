# COMP 3105 Fall 2025 Assignment 2
# Raphael Mercier & Patrick Wu

from cvxopt import matrix, solvers
from scipy.optimize import minimize
import numpy as np  
import pandas as pd
import os 

#Q1 A TODO: Verify that this function retuns the correct value, in particular the margin calculation
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

#Q1 B
def minHinge(X, y, lamb, stablizer=1e-5):

    # Get # of elements and "features" and then get the diagonal matrix diag Y
    n = X.shape[0]
    d = X.shape[1]
    Y = np.diag(y.flatten())

    #Create the q vector consisting of (d+1) zeroes and n ones
    q1 = np.zeros(d+1)
    q2 = np.ones(n)
    q = np.concatenate([q1, q2])

    #Solve the first constraint parts G1 and h1: G11 = 0n×d , G12 = 0n×1, G13 = −In and h1 = 0n
    G11 = np.zeros((n, d))
    G12 = np.zeros((n, 1))
    G13 = -np.eye(n)
    h1 = np.zeros((n, 1))
    G1 = np.concatenate([G11, G12, G13], axis = 1)

    #Solve the second constraint parts G2 and h2: G21 = −∆(y)X , G22 = −∆(y)1n = −y, G23 = −In and h2 = −1n
    G21 = -Y @ X 
    G22 = -y
    G23 = -np.eye(n)
    h2 = -np.ones((n, 1))
    G2 = np.concatenate([G21, G22, G23], axis = 1)

    # put together G and h
    G = np.concatenate([G1, G2], axis = 0)
    h = np.concatenate([h1, h2], axis = 0)

    # put together P where P11 = λId , everything else is 0
    P11 = lamb * np.eye(d)
    P12and3 = np.zeros((d, n + 1))
    P1 = np.concatenate([P11, P12and3], axis = 1)
    P2 = np.zeros((1, n + d + 1))
    P3 = np.zeros((n, n + d + 1))

    # Add small positive stabilizer to the diagonal to ensure numerical stability for the P matrix
    P = np.concatenate([P1, P2, P3], axis = 0)
    P = P + stablizer * np.eye(n+d+1)

    # Convert P, q, G and H to cvxopt matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    # solve
    solution = solvers.qp(P, q, G, h)

    # convert cvxopt matrix to array
    arr = np.array(solution['x']).flatten()

    # extract w (first d elements) and w0 (the d + 1 th element in the array)
    return arr[:d], arr[d]

