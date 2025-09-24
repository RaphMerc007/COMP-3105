# COMP 3105 Fall 2025 Assignment 1
# Raphael Mercier & Patrick Wu
# Machine Learning Optimization Algorithms Implementation

from math import ceil, floor
from cvxopt import matrix, solvers
from scipy.optimize import minimize
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

    #compare minimizeL2 and find_opt
    # print("weights from minimizeL2:", w_L2.flatten()) 
    
    # w_opt = find_opt(linearRegL2Obj, linearRegL2Grad, Xtrain, ytrain)

    # print("weights from find_opt:", w_opt.flatten()) 

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

# Returns the objective value for the L2 objective 
# function 1/2n ||Xw - y||^2 = 1/2n(Xw - y)ᵀ(Xw - y)
def linearRegL2Obj(w, X, y):

  # flattens y to be a 1D array to ensure correct matrix operations
  y_flat = y.flatten()
  residuals = X @ w - y_flat
  return (1/(2*y_flat.shape[0])) * residuals.T @ residuals

# Returns a d x 1 vector representing the gradient, using the formula
# 1/n Xᵀ(Xw - y)
def linearRegL2Grad(w, X, y):
  
  # flattens y to be a 1D array to ensure correct matrix operations
  y_flat = y.flatten()
  residuals = (X @ w) - y_flat
  return (1/y_flat.shape[0]) * X.T @ residuals

# Uses the Indirect approach to find the optimal w vector weights
def find_opt(obj_func, grad_func, X, y):
  d = X.shape[1]
  # Initialize a random 1-D array of parameters of size d
  w_0 = np.random.randn(d) 
  
  # Define an objective function `func` that takes a single argument (w)
  func = lambda w: obj_func(w, X, y)

  # Define a gradient function `gd` that takes a single argument (w)
  gd = lambda w: grad_func(w, X, y)

  return minimize(func, w_0, jac=gd)['x'][:, None]

def logisticRegObj(w, X, y):
  # TODO: Implement logistic regression objective

  y_flat = y.flatten()
  n = y_flat.shape[0]
  ones_vector = np.ones(n)

  # Solve sigmoid function σ(Xw) 
  # sigmoid = 1 / (1 + np.exp(-(X @ w)))
  sigmoid = np.exp(-np.logaddexp(0, - (X @ w)))

  # -yT log(sigma (Xw)) - (1n -y)T log(1n - sigma(Xw))
  term1 = -y_flat.T @ np.log(sigmoid)
  term2 = (ones_vector - y_flat).T @ np.log(ones_vector - sigmoid)

  return (1/n) * (term1 - term2)


def logisticRegGrad(w, X, y):
  # TODO: Implement logistic regression gradient

  y_flat = y.flatten()
  sigmoid = np.exp(-np.logaddexp(0, - (X @ w)))

  return (1/y_flat.shape[0]) * X.T @ (sigmoid - y_flat)

def synClsExperiments():
  def genData(n_points, dim1, dim2):
    '''
    This function generate synthetic data
    '''
    c0 = np.ones([1, dim1]) # class 0 center
    c1 = -np.ones([1, dim1]) # class 1 center
    X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
    X0[:, :dim1] += c0
    X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
    X1[:, :dim1] += c1
    X = np.concatenate((X0, X1), axis=0)
    X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
    y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
    return X, y

  def load_csv_data(train_path, test_path):
    """
    Load data from CSV files for classification
    """
    import pandas as pd

    # Load training data
    train_df = pd.read_csv(train_path)
    Xtrain = train_df[['x1', 'x2']].values  
    ytrain = train_df[['y']].values        

    # Add bias term (column of ones) to Xtrain
    Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)

    # Load test data
    test_df = pd.read_csv(test_path)
    Xtest = test_df[['x1', 'x2']].values  
    ytest = test_df[['y']].values         

    # Add bias term (column of ones) to Xtest
    Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    return Xtrain, ytrain, Xtest, ytest

  def runClsExp(m=100, dim1=2, dim2=2):
    '''
    Run classification experiment with the specified arguments
    '''
    n_test = 1000
    Xtrain, ytrain = genData(m, dim1, dim2)
    Xtest, ytest = genData(n_test, dim1, dim2)

    # Using the toy data that was given
    # Xtrain, ytrain, Xtest, ytest  = load_csv_data("./toy_data/classification_train.csv","./toy_data/classification_test.csv")

    w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

    # print estimated w weights (only use for debug, this spams the console lmao)
    # print("weights from find_opt ", w_logit.flatten())

    # ytrain_hat = sigmoid(XTrain^T * w_logit) 
    # Compute predicted labels of the training points
    ytrain_hat = np.exp(-np.logaddexp(0, -(Xtrain @ w_logit)))

    # train_acc = Compute the accuarcy of the training set
    # Convert ytrain_hat values to 1 if probability >= 0, else 0
    ytrain_hat_bool = (ytrain_hat >= 0.5).astype(int)
    train_acc = np.mean(ytrain_hat_bool.flatten() == ytrain.flatten())

    # ytest_hat = Compute predicted labels of the test points
    ytest_hat = np.exp(-np.logaddexp(0, -(Xtest @ w_logit)))

    # test_acc = Compute the accuarcy of the test set
    ytest_hat_bool = (ytest_hat >= 0.5).astype(int)
    test_acc = np.mean(ytest_hat_bool.flatten() == ytest.flatten())

    return train_acc, test_acc
  
  n_runs = 100
  train_acc = np.zeros([n_runs, 4, 3])
  test_acc = np.zeros([n_runs, 4, 3])
  # Change the following random seed to one of your student IDs
  np.random.seed(101258669)
  for r in range(n_runs):
    for i, m in enumerate((10, 50, 100, 200)):
      train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
    for i, dim1 in enumerate((1, 2, 4, 8)):
      train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
    for i, dim2 in enumerate((1, 2, 4, 8)):
      train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)
  
  # compute the average accuracies over runs, across dimension 0 (r, the # of runs)
  train_avg_acc = np.mean(train_acc, axis=0)
  test_avg_acc = np.mean(test_acc, axis=0)

  #return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable
  return train_avg_acc, test_avg_acc

# def preprocessBCW(dataset_folder):

#   # TODO: implement function 

#   pass

# def runBCW(dataset_folder):
#   X, y = preprocessBCW(dataset_folder)
#   n, d = X.shape
#   X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
#   n_runs = 100
#   train_acc = np.zeros([n_runs])
#   test_acc = np.zeros([n_runs])
#   # TODO: Change the following random seed to one of your student IDs
#   np.random.seed(42)
#   for r in range(n_runs):
#   # TODO: Randomly partition the dataset into two parts (50%
#   # training and 50% test)
#   w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)
#   # TODO: Evaluate the model's accuracy on the training
#   # data. Save it to `train_acc`
#   # TODO: Evaluate the model's accuracy on the test
#   # data. Save it to `test_acc`
#   # TODO: compute the average accuracies over runs
#   # TODO: return two variables: the average training accuracy and average test accuracy

#   pass