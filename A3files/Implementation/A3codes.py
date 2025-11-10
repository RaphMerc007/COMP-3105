# COMP 3105 Fall 2025 Assignment 3
# Raphael Mercier & Patrick Wu

from cvxopt import matrix, solvers
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigh
import numpy as np  
import pandas as pd
from A3helpers import convertToOneHot
from A3helpers import generateData, augmentX, plotImgs

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



# Q2 a
def PCA(X, k):
  _, d = X.shape

  # get mean of each column
  mean = np.mean(X, axis = 0)
  # center the data
  X_centered = X - mean

  # get the last k largest eigenvalues and eigenvectors of X^t X
  eigvals, eigvecs = eigh(X_centered.T @ X_centered, subset_by_index = [d - k, d-1])

  # sort the eigenvalues in descending order
  idx = np.argsort(eigvals)[::-1]
  eigvals = eigvals[idx]
  eigvecs = eigvecs[:, idx]

  # make sure we only get the top k eigenvectors
  principal_components = eigvecs[:, :k]

  # project the data onto the principal components
  X_transformed = X_centered @ principal_components


  return X_transformed

# Q2 b
def projPCA(Xtest, mu, U):
  mu_row = mu.reshape(1, -1) 
  
  return (Xtest - mu_row) @ U 

# Q2 c
def kernelPCA(X, k, kernel_func):
  n, _ = X.shape
  X = X.astype(float)

  # get the kernel matrix
  K = kernel_func(X, X)

  # SLOW
  # K_norm = K - 1/n * np.ones((n, n)) @ K - 1/n * K @ np.ones((n, n)) + 1/n**2 * np.ones((n, n)) @ K @ np.ones((n, n))
  
  # FAST broadcasting method
  K_row_mean = np.mean(K, axis=1, keepdims=True)  # Column vector
  K_col_mean = np.mean(K, axis=0, keepdims=True)  # Row vector
  K_mean = np.mean(K)
  K_norm = K - K_row_mean - K_col_mean + K_mean

  # get the last k largest eigenvalues and eigenvectors of K_norm
  eigvals, eigvecs = eigh(K_norm, subset_by_index = [n - k, n-1])

  # sort the eigenvalues in descending order
  idx = np.argsort(eigvals)[::-1]
  eigvals = eigvals[idx]
  eigvecs = eigvecs[:, idx]

  # After getting eigvecs and eigvals, normalize:
  alphas = eigvecs[:, :k] / np.sqrt(eigvals[:k])  # Divide by sqrt(eigenvalue)
  return alphas.T  # (k, n)

# Q2 d
def projKernelPCA(Xtest, Xtrain, kernel_func, A):
  # Compute kernel matrices
  K_test = kernel_func(Xtest, Xtrain) 
  K_train = kernel_func(Xtrain, Xtrain) 
  
  # Center the kernel matrix using broadcasting
  K_train_col_mean = np.mean(K_train, axis=0, keepdims=True)  
  K_test_row_mean = np.mean(K_test, axis=1, keepdims=True)   
  K_train_mean = np.mean(K_train)                      
  
  Kete_tr = K_test - K_train_col_mean - K_test_row_mean + K_train_mean
  
  return Kete_tr @ A.T 

# Q2 e
def synClsExperimentsPCA():
  n_runs = 100
  n_train = 128
  n_test = 1000
  dim_list = [1, 2]
  gen_model_list = [1, 2]
  train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
  test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
  np.random.seed(91)
  for r in range(n_runs):
    for i, k in enumerate(dim_list):
      for j, gen_model in enumerate(gen_model_list):
        Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
        Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)

        # Get mean and compute PCA
        mu = np.mean(Xtrain, axis=0)
        Xtrain_centered = Xtrain - mu
        
        # Get principal components
        eigvals, eigvecs = eigh(Xtrain_centered.T @ Xtrain_centered, subset_by_index=[Xtrain.shape[1] - k, Xtrain.shape[1] - 1])
        idx = np.argsort(eigvals)[::-1]
        U = eigvecs[:, idx][:, :k]  # (d, k) principal components

        Xtrain_proj = projPCA(Xtrain, mu, U) 
        Xtest_proj = projPCA(Xtest, mu, U)

        Xtrain_proj = augmentX(Xtrain_proj) # add augmentation
        Xtest_proj = augmentX(Xtest_proj)
        W = minMulDev(Xtrain_proj, Ytrain) # from Q1

        Yhat = classify(Xtrain_proj, W) # from Q1
        train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1

        Yhat = classify(Xtest_proj, W)
        test_acc[i, j, r] = calculateAcc(Yhat, Ytest)

  # compute the average accuracies over runs
  train_acc = np.mean(train_acc, axis = 2)
  test_acc = np.mean(test_acc, axis = 2)

  # return 2-by-2 train accuracy and 2-by-2 test accuracy
  return train_acc, test_acc

# Q3 a
def kmeans(X, k, max_iter=1000):
  n, d = X.shape
  assert max_iter > 0 and k < n
  # TODO: Choose k random points from X as initial centers
  U = 
  for i in range(max_iter):
    # TODO: Compute pairwise distance between X and U
    D = 
    
    # TODO: Find the new cluster assignments
    Y =
    
    old_U = U
    
    # TODO: Update cluster centers
    U = 

    if np.allclose(old_U, U):
      break
  obj_val = (0.5 / n) * np.sum(D.min(axis=1))
  return Y, U, obj_val

# Q3 b
def repeatKmeans(X, k, n_runs=100):
  best_obj_val = float('inf')
  for r in range(n_runs):
    Y, U, obj_val = kmeans(X, k)
    # TODO: Compare obj_val with best_obj_val. If it is lower,
    # then record the current Y, U and update best_obj_val
  
  # TODO: Return the best Y, U and best_obj_val


# Q3 c
def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
  n, d = X.shape
  assert max_iter > 0 and k < n
  K = kernel_func(X, X)
  Y = init_Y
  for i in range(max_iter):
    # TODO: Compute pairwise distance matrix
    D = 

    old_Y = Y
    
    # TODO: Find the new cluster assignments
    Y = 
    
    if np.allclose(old_Y, Y):
      break
  
  obj_val = (0.5 / n) * np.sum(D.min(axis=1))
  return Y, obj_val