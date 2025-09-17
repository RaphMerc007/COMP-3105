# COMP 3105 Fall 2025 Assignment 1
# Raphael Mercier & Patrick
# Machine Learning Optimization Algorithms Implementation

import numpy as np
from scipy.optimize import minimize


def minimizeL2(X, y):
  # This follows the equation:
  # (X^T * X)^−1 * X^T * y
  return np.linalg.inv(X.T @ X) @ X.T @ y


def minimizeLinf(X, y):
  # TODO: Implement L∞ regression
  pass


def logisticRegObj(w, X, y):
  # TODO: Implement logistic regression objective
  pass


def logisticRegGrad(w, X, y):
  # TODO: Implement logistic regression gradient
  pass


def find_opt(obj_func, grad_func, X, y):
  # TODO: Implement optimization routine
  pass