# COMP 3105 Fall 2025 Assignment 1
# Raphael Mercier & Patrick
# Machine Learning Optimization Algorithms Implementation

from cvxopt import matrix, solvers
import numpy as np
from scipy.optimize import minimize


def minimizeL2(X, y):
  # This follows the equation:
  # (X^T * X)^−1 * X^T * y
  return np.linalg.inv(X.T @ X) @ X.T @ y


def minimizeLinf(X, y):
  # Define shape of matrix
  n = X.shape[0]
  d = X.shape[1]
  
  # c^⊤ u = δ
  c = np.zeros[1, d+1]
  c[d] = [1]

  # G(1)u ⪯ h(1) ⇐⇒ δ ≥ 0
  h1 = [0]
  G1 = np.zeros[d+1, 1]
  G1[d] = -1

  # G(2)u ⪯ h(2) ⇐⇒ Xw − y ⪯ δ·1n
  h2 = y
  G2 = np.concatenate([X, -1 * np.ones(n, 1)], axis=1)

  # G(3)u ⪯ h(3) ⇐⇒ y − Xw ⪯ δ·1n
  h3 = -y
  G3 = np.concatenate([-X, -1 * np.ones(n, 1)], axis=1)

  # Combine parts of G and h together
  G = np.concatenate([G1, G2, G3], axis=0)
  h = np.concatenate([h1, h2, h3], axis=0)

  # convert to cvxopt matrix
  c = matrix(c)
  G = matrix(G)
  h = matrix(h)

  # TODO: Need to uncomment this before submition
  # solvers.options['show progress'] = False

  return solvers.lp(c,G,h)

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
    Xtrain, ytrain = genData(n_train, is_training=True)
    Xtest, ytest = genData(n_test, is_training=False)
    w_L2 = minimizeL2(Xtrain, ytrain)
    w_Linf = minimizeLinf(Xtrain, ytrain)
    # TODO: Evaluate the two models' performance (for each model, calculate the L2 and L infinity losses on the training data). Save them to `train_loss`
    
    # TODO: Evaluate the two models' performance (for each model, calculate the L2 and L infinity losses on the test data). Save them to `test_loss`
  
  # TODO: compute the average losses over runs
  
  # TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable

def logisticRegObj(w, X, y):
  # TODO: Implement logistic regression objective
  pass


def logisticRegGrad(w, X, y):
  # TODO: Implement logistic regression gradient
  pass


def find_opt(obj_func, grad_func, X, y):
  # TODO: Implement optimization routine
  pass