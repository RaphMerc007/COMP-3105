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


def logisticRegObj(w, X, y):
  # TODO: Implement logistic regression objective
  pass


def logisticRegGrad(w, X, y):
  # TODO: Implement logistic regression gradient
  pass


def find_opt(obj_func, grad_func, X, y):
  # TODO: Implement optimization routine
  pass