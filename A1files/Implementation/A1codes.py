# COMP 3105 Fall 2025 Assignment 1
# Raphael Mercier & Patrick Wu
# Machine Learning Optimization Algorithms Implementation

from math import ceil, floor
from cvxopt import matrix, solvers
import numpy as np  
import pandas as pd
import os 


def minimizeL2(X, y):
  # This follows the equation:
  # (X^T * X)^−1 * X^T * y
  return np.linalg.inv(X.T @ X) @ X.T @ y

def minimizeLinf(X, y):
  # Define shape of matrix
  n, d = X.shape
  
  # c^⊤ u = δ
  c = np.zeros([d+1])
  c[d] = 1

  # G(1)u ⪯ h(1) ⇐⇒ δ ≥ 0
  h1 = [0]
  G1 = np.zeros([1, d+1])
  G1[0, d] = -1

  # G(2)u ⪯ h(2) ⇐⇒ Xw − y ⪯ δ·1n
  h2 = y.flatten()
  G2 = np.concatenate([X, -1 * np.ones([n, 1])], axis=1)

  # G(3)u ⪯ h(3) ⇐⇒ y − Xw ⪯ δ·1n
  h3 = (-y).flatten()
  G3 = np.concatenate([-X, -1 * np.ones([n, 1])], axis=1)

  # Combine parts of G and h together
  G = np.concatenate([G1, G2, G3], axis=0)
  h = np.concatenate([h1, h2, h3], axis=0)

  # convert to cvxopt matrix
  c = matrix(c)
  G = matrix(G)
  h = matrix(h)

  solvers.options['show_progress'] = False

  sol = solvers.lp(c,G,h)
  return np.array(sol['x'])[:d]

def synRegExperiments():
  def genData(n_points, is_training=False):
    '''
    This function generate synthetic data
    '''
    X = np.random.randn(n_points, d) # input matrix
    X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
    y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
    if is_training:
      y[0] *= -0.1
    return X, y

  def load_csv_data(train_path, test_path):
    """
    Load data from CSV files for regression
    """
    import pandas as pd
    
    # Load training data
    train_df = pd.read_csv(train_path)
    Xtrain = train_df[['x']].values
    ytrain = train_df[['y']].values
    
    # Add bias term (column of ones) to Xtrain
    Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    
    # Load test data
    test_df = pd.read_csv(test_path)
    Xtest = test_df[['x']].values
    ytest = test_df[['y']].values
    
    # Add bias term (column of ones) to Xtest
    Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    return Xtrain, ytrain, Xtest, ytest


  n_runs = 100
  n_train = 30
  n_test = 1000
  d = 5
  noise = 0.2
  train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
  test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
  np.random.seed(101306865)

  for r in range(n_runs):
    w_true = np.random.randn(d + 1, 1)
    
    # Using genData
    Xtrain, ytrain = genData(n_train, is_training=True)
    Xtest, ytest = genData(n_test, is_training=False)

    # Using the toy data that was given
    # Xtrain, ytrain, Xtest, ytest  = load_csv_data("./toy_data/regression_train.csv","./toy_data/regression_test.csv")

    w_L2 = minimizeL2(Xtrain, ytrain)
    w_Linf = minimizeLinf(Xtrain, ytrain)
  
    # Get train data loss
    # Get Xw for different models
    train_pred_L2 = Xtrain @ w_L2
    train_pred_Linf = Xtrain @ w_Linf

    # L2 model, L2 loss
    train_loss[r, 0, 0] = np.mean(0.5 * ((ytrain - train_pred_L2)**2))
    # L2 model, Linf loss
    train_loss[r, 0, 1] = np.max(np.abs(ytrain - train_pred_L2))
    # Linf model, L2 loss
    train_loss[r, 1, 0] = np.mean(0.5 * ((ytrain - train_pred_Linf)**2))
    # Linf model, Linf loss
    train_loss[r, 1, 1] = np.max(np.abs(ytrain - train_pred_Linf))


    # Get test data loss
    # Get Xw for different models
    test_pred_L2 = Xtest @ w_L2
    test_pred_Linf = Xtest @ w_Linf

    # L2 model, L2 loss
    test_loss[r, 0, 0] = np.mean(0.5 * ((ytest - test_pred_L2)**2))
    # L2 model, Linf loss
    test_loss[r, 0, 1] = np.max(np.abs(ytest - test_pred_L2))
    # Linf model, L2 loss
    test_loss[r, 1, 0] = np.mean(0.5 * ((ytest - test_pred_Linf)**2))
    # Linf model, Linf loss
    test_loss[r, 1, 1] = np.max(np.abs(ytest - test_pred_Linf))
  
  
  # compute average losses for training data
  train_avg_loss = np.zeros([2,2])
  train_avg_loss[0,0] = np.sum(train_loss[:,0,0])/n_runs
  train_avg_loss[0,1] = np.sum(train_loss[:,0,1])/n_runs
  train_avg_loss[1,0] = np.sum(train_loss[:,1,0])/n_runs
  train_avg_loss[1,1] = np.sum(train_loss[:,1,1])/n_runs
  
  # compute average losses for test data
  test_avg_loss = np.zeros([2,2])
  test_avg_loss[0,0] = np.sum(test_loss[:,0,0])/n_runs
  test_avg_loss[0,1] = np.sum(test_loss[:,0,1])/n_runs
  test_avg_loss[1,0] = np.sum(test_loss[:,1,0])/n_runs
  test_avg_loss[1,1] = np.sum(test_loss[:,1,1])/n_runs

  return train_avg_loss, test_avg_loss

def runCCS(dataset_folder):
  X, y = preprocessCCS(dataset_folder)
  n, d = X.shape
  X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
  n_runs = 100
  train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
  test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
  np.random.seed(101306865)

  for r in range(n_runs):
    # distributing data accross train and test data sets
    train_data = np.zeros([floor(n/2+1), d+2])
    test_data = np.zeros([ceil(n/2+1), d+2])
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    j = 0
    for i in range(n):
      if i <= n/2:
        train_data[i, :-1] = X[indexes[i], :]
        train_data[i, -1] = y[indexes[i]]
      else:
        test_data[j, :-1] = X[indexes[i], :]
        test_data[j, -1] = y[indexes[i]]
        j+=1

    Xtrain, ytrain = train_data[:, :-1], train_data[:, -1]
    Xtest, ytest = test_data[:, :-1], test_data[:, -1]

    w_L2 = minimizeL2(Xtrain, ytrain)
    w_Linf = minimizeLinf(Xtrain, ytrain)
  
    # Get train data loss
    # Get Xw for different models
    train_pred_L2 = Xtrain @ w_L2
    train_pred_Linf = Xtrain @ w_Linf

    # L2 model, L2 loss
    train_loss[r, 0, 0] = np.mean(0.5 * ((ytrain - train_pred_L2)**2))
    # L2 model, Linf loss
    train_loss[r, 0, 1] = np.max(np.abs(ytrain - train_pred_L2))
    # Linf model, L2 loss
    train_loss[r, 1, 0] = np.mean(0.5 * ((ytrain - train_pred_Linf)**2))
    # Linf model, Linf loss
    train_loss[r, 1, 1] = np.max(np.abs(ytrain - train_pred_Linf))


    # Get test data loss
    # Get Xw for different models
    test_pred_L2 = Xtest @ w_L2
    test_pred_Linf = Xtest @ w_Linf

    # L2 model, L2 loss
    test_loss[r, 0, 0] = np.mean(0.5 * ((ytest - test_pred_L2)**2))
    # L2 model, Linf loss
    test_loss[r, 0, 1] = np.max(np.abs(ytest - test_pred_L2))
    # Linf model, L2 loss
    test_loss[r, 1, 0] = np.mean(0.5 * ((ytest - test_pred_Linf)**2))
    # Linf model, Linf loss
    test_loss[r, 1, 1] = np.max(np.abs(ytest - test_pred_Linf))
    pass

  # compute average losses for training data
  train_avg_loss = np.zeros([2,2])
  train_avg_loss[0,0] = np.sum(train_loss[:,0,0])/n_runs
  train_avg_loss[0,1] = np.sum(train_loss[:,0,1])/n_runs
  train_avg_loss[1,0] = np.sum(train_loss[:,1,0])/n_runs
  train_avg_loss[1,1] = np.sum(train_loss[:,1,1])/n_runs
  
  # compute average losses for test data
  test_avg_loss = np.zeros([2,2])
  test_avg_loss[0,0] = np.sum(test_loss[:,0,0])/n_runs
  test_avg_loss[0,1] = np.sum(test_loss[:,0,1])/n_runs
  test_avg_loss[1,0] = np.sum(test_loss[:,1,0])/n_runs
  test_avg_loss[1,1] = np.sum(test_loss[:,1,1])/n_runs

  return train_avg_loss, test_avg_loss


def preprocessCCS(dataset_folder):
  # Construct path to the Excel file
  excel_path = os.path.join(dataset_folder, 'Concrete_Data.xls')
  
  # Load the Excel file
  df = pd.read_excel(excel_path)

  # Extract features (all columns except the last one)
  X = df.iloc[:, :-1].values  # n x d matrix
  # Extract target variable (last column)
  y = df.iloc[:, -1].values.reshape(-1, 1)  # n x 1 vector

  return X, y

def logisticRegObj(w, X, y):
  # TODO: Implement logistic regression objective
  pass


def logisticRegGrad(w, X, y):
  # TODO: Implement logistic regression gradient
  pass


def find_opt(obj_func, grad_func, X, y):
  # TODO: Implement optimization routine
  pass