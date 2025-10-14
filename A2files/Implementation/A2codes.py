# COMP 3105 Fall 2025 Assignment 2
# Raphael Mercier & Patrick Wu

from cvxopt import matrix, solvers
from scipy.optimize import minimize
import numpy as np  
import pandas as pd
from A2helpers import generateData, linearKernel, polyKernel, gaussKernel

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
        margins = y.flatten() * (X @ w + w0)

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
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h)

    # convert cvxopt matrix to array
    arr = np.array(solution['x']).flatten()

    # extract w (first d elements) and w0 (the d + 1 th element in the array)
    return arr[:d], arr[d]

# Q1 C
def classify(Xtest, w, w0):
   # returns the m x 1 prediction vector y-hat = sign(Xtest x w + w0) given an (m x d) test matrix Xtest
   return np.sign(Xtest @ w + w0)

# Q1 D
def synExperimentsRegularize():
  n_runs = 100
  n_train = 100
  n_test = 1000
  lamb_list = [0.001, 0.01, 0.1, 1.]
  gen_model_list = [1, 2, 3]
  train_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
  test_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
  train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
  test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
  np.random.seed(125)
  for r in range(n_runs):
    for i, lamb in enumerate(lamb_list):
      for j, gen_model in enumerate(gen_model_list):
        Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
        Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

        w, w0 = minExpLinear(Xtrain, ytrain, lamb)

        # Get the y hat predictions for weights learned using the minExpLinear loss
        yHatTrain = classify(Xtrain, w, w0)
        yHatTest = classify(Xtest, w, w0)

        # Computes accuracy on training and test sets respectively for the minExpLinear loss (flatten to ensure both arrays
        # are the same and can be compared properly)
        train_acc_explinear[i, j, r] = np.mean(yHatTrain.flatten() == ytrain.flatten())
        test_acc_explinear[i, j, r] = np.mean(yHatTest.flatten() == ytest.flatten())
        
        w, w0 = minHinge(Xtrain, ytrain, lamb)

        # Get the y hat predictions for weights learned using the minHinge loss
        yHatTrain = classify(Xtrain, w, w0)
        yHatTest = classify(Xtest, w, w0)

        #Computes accuracy on training and test sets respectively for the minHinge loss
        train_acc_hinge[i, j, r] = np.mean(yHatTrain.flatten() == ytrain.flatten())
        test_acc_hinge[i, j, r] = np.mean(yHatTest.flatten() == ytest.flatten())

  # compute the average accuracies over runs
  trainAccExpLinear = np.mean(train_acc_explinear, axis = 2)
  trainAccHinge = np.mean(train_acc_hinge, axis = 2)
  testAccExpLinear = np.mean(test_acc_explinear, axis = 2)
  testAccHinge = np.mean(test_acc_hinge, axis = 2)

  # combine accuracies (explinear and hinge)
  trainAcc = np.concatenate((trainAccExpLinear, trainAccHinge), axis = 1)
  testAcc = np.concatenate((testAccExpLinear, testAccHinge), axis = 1)

  # return 4-by-6 train accuracy and 4-by-6 test accuracy
  return trainAcc, testAcc


# q2 A
def adjExpLinear(X, y, lamb, kernel_func):

  # compute the kernel matrix
  K = kernel_func(X,X)
  n, d = X.shape

  # define the loss function for the regularized ExpLinear loss
  def loss(params):
    # extract the first n elements from the params representing weights, and remaining element as the intercept
    a = params[:-1]
    a0 = params[-1]

    # compute margins for all data points at once (vectorized)
    margins = y.flatten() * (K @ a + a0)

    # calculate loss terms
    sum = np.sum(np.maximum(0, -margins) + np.exp(np.minimum(0, -margins)))
    regularization = (lamb / 2) * a.T @ K @ a
    
    return sum + regularization

  # initialize a zero (n+1) x 1 vector storing the initial guess values
  x0 = np.zeros(n+1)

  # solve
  result = minimize(loss, x0)

  # extract results from the scipy minimize function
  a_star = result.x[:-1]
  a0_star = result.x[-1]
  
  return a_star, a0_star

# q2 B
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
  n,d = X.shape
  K = kernel_func(X,X)
  Y = np.diag(y.flatten())

  #Create the q vector consisting of (n+1) zeroes and n ones
  q1 = np.zeros(n+1)
  q2 = np.ones(n)
  q = np.concatenate([q1, q2])

  #Solve the first constraint parts G1 and h1: G11 = 0n×n , G12 = 0n×1, G13 = −In and h1 = 0n
  G11 = np.zeros((n, n))
  G12 = np.zeros((n, 1))
  G13 = -np.eye(n)
  h1 = np.zeros((n, 1))
  G1 = np.concatenate([G11, G12, G13], axis = 1)

  #Solve the second constraint parts G2 and h2: G21 = −∆(y)K , G22 = −∆(y)1n = −y, G23 = −In and h2 = −1n
  G21 = -Y @ K
  G22 = -y
  G23 = -np.eye(n)
  h2 = -np.ones((n, 1))
  G2 = np.concatenate([G21, G22, G23], axis = 1)

  # put together G and h
  G = np.concatenate([G1, G2], axis = 0)
  h = np.concatenate([h1, h2], axis = 0)

  # put together P where P11 = λK , everything else is 0
  P11 = lamb * K
  P12and3 = np.zeros((n, n + 1))
  P1 = np.concatenate([P11, P12and3], axis = 1)
  P2 = np.zeros((1, 2*n + 1))
  P3 = np.zeros((n, 2*n + 1))

  # Add small positive stabilizer to the diagonal to ensure numerical stability for the P matrix
  P = np.concatenate([P1, P2, P3], axis = 0)
  P = P + stabilizer * np.eye(2*n+1)

  # Convert P, q, G and H to cvxopt matrices
  P = matrix(P)
  q = matrix(q)
  G = matrix(G)
  h = matrix(h)

  # solve
  solvers.options['show_progress'] = False
  solution = solvers.qp(P, q, G, h)

  # convert cvxopt matrix to array
  arr = np.array(solution['x']).flatten()

  # extract a (first n elements) and a0 (the n + 1 th element in the array)
  return arr[:n], arr[n]

# q2 C
def adjClassify(Xtest, a, a0, X, kernel_func):
  # returns the m x 1 prediction vector y-hat = sign(K(Xtest, X) @ a + a0) given an (m x d) test matrix Xtest
  return np.sign(kernel_func(Xtest, X) @ a + a0)

# q2 D
def synExperimentsKernel():
  n_runs = 10
  n_train = 100
  n_test = 1000
  lamb = 0.001
  kernel_list = [linearKernel,
                lambda X1, X2: polyKernel(X1, X2, 2),
                lambda X1, X2: polyKernel(X1, X2, 3),
                lambda X1, X2: gaussKernel(X1, X2, 1.0),
                lambda X1, X2: gaussKernel(X1, X2, 0.5)]
  gen_model_list = [1, 2, 3]
  train_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
  test_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
  train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
  test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
  np.random.seed(125)
  for r in range(n_runs):
    for i, kernel in enumerate(kernel_list):
      for j, gen_model in enumerate(gen_model_list):
        Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
        Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

        
        a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel)

        # Get the y hat predictions for weights learned using the minExpLinear loss
        yHatTrain = adjClassify(Xtrain, a, a0, Xtrain, kernel)
        yHatTest = adjClassify(Xtest, a, a0, Xtrain, kernel)

        # Computes accuracy on training and test sets respectively for the minExpLinear loss (flatten to ensure both arrays
        # are the same and can be compared properly)
        train_acc_explinear[i, j, r] = np.mean(yHatTrain.flatten() == ytrain.flatten())
        test_acc_explinear[i, j, r] = np.mean(yHatTest.flatten() == ytest.flatten())


        a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)

        # Get the y hat predictions for weights learned using the minHinge loss
        yHatTrain = adjClassify(Xtrain, a, a0, Xtrain, kernel)
        yHatTest = adjClassify(Xtest, a, a0, Xtrain, kernel)

        #Computes accuracy on training and test sets respectively for the minHinge loss
        train_acc_hinge[i, j, r] = np.mean(yHatTrain.flatten() == ytrain.flatten())
        test_acc_hinge[i, j, r] = np.mean(yHatTest.flatten() == ytest.flatten())

  # compute the average accuracies over runs
  trainAccExpLinear = np.mean(train_acc_explinear, axis = 2)
  trainAccHinge = np.mean(train_acc_hinge, axis = 2)
  testAccExpLinear = np.mean(test_acc_explinear, axis = 2)
  testAccHinge = np.mean(test_acc_hinge, axis = 2)

  # combine accuracies (explinear and hinge)
  trainAcc = np.concatenate((trainAccExpLinear, trainAccHinge), axis = 1)
  testAcc = np.concatenate((testAccExpLinear, testAccHinge), axis = 1)

  return trainAcc, testAcc





# q3 A
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
  n,d = X.shape
  K = kernel_func(X, X)
  Y = np.diag(y.flatten())

  # put together P where P = 1/λ * Y @ K @ Y
  P = 1/lamb * Y @ K @ Y
  P = P + stabilizer * np.eye(n)


  # create the q vector consisting of n ones
  q = -1 * np.ones((n, 1))

  # create the G1 matrix consisting of -In
  G1 = -1 * np.eye(n)
  # create the G2 matrix consisting of In
  G2 = np.eye(n)
  # put together G
  G = np.concatenate([G1, G2], axis = 0)

  # create the h1 vector consisting of 0n
  h1 = np.zeros((n, 1))
  # create the h2 vector consisting of 1n
  h2 = np.ones((n, 1))
  # put together h
  h = np.concatenate([h1, h2], axis = 0)

  # create the A matrix consisting of y.T
  A = y.T
  # create the b matrix consisting of 1
  b = np.ones((1, 1))

  # convert to cvxopt matrices
  P = matrix(P)
  q = matrix(q)
  G = matrix(G)
  h = matrix(h)
  A = matrix(A)
  b = matrix(b)

  # solve
  solvers.options['show_progress'] = False
  solution = solvers.qp(P, q, G, h, A, b)

  # convert to array
  a = np.array(solution['x']).flatten()

  # Compute b using the support vector closest to 0.5
  # create the support vectors
  support_vectors = (a > 1e-5) & (a < 1 - 1e-5)

  # find distances to 0.5
  distances = np.abs(a - 0.5)
  distances[~support_vectors] = np.inf  # Ignore non-support vectors

  # pick the best one
  i = np.argmin(distances)

  # Compute b = y_i - (1/λ) k_i^T Δ(y)α*
  k_i = K[i, :]
  b_offset = y[i, 0] - (1/lamb) * k_i @ Y @ a

  return a, b_offset

# q3 B
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
  Y = np.diag(y.flatten())
  # returns the m x 1 prediction vector y-hat = sign(1/λ * K(Xtest, X) @ Y @ a + b) given an (m x d) test matrix Xtest
  return np.sign(1/lamb * kernel_func(Xtest, X) @ Y @ a + b)


# q3 C
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
  train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
  X = train_data[:, 1:] / 255.
  y = train_data[:, 0][:, None]
  y[y == 4] = -1
  y[y == 9] = 1
  cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])
  np.random.seed(125)
  length = X.shape[0]//k
  
  for i, lamb in enumerate(lamb_list):
    for j, kernel_func in enumerate(kernel_list):
      for l in range(k):
        # determine bounds for the validation set
        start = l * length
        end = start + length
        
        # determine the validation set
        Xval = X[start:end,:]
        yval = y[start:end,:]

        # determine the training set
        Xtrain = np.concatenate([X[:start,:], X[end:,:]], axis = 0)
        ytrain = np.concatenate([y[:start,:], y[end:,:]], axis = 0)

        # determine the weights and bias
        a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)

        # determine the predictions for the validation set
        yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
        
        # determine the accuracy for the validation set
        cv_acc[l, i, j] = np.mean(yhat.flatten() == yval.flatten())
        
  # determine the average accuracy for the validation set
  cv_acc = np.mean(cv_acc, axis = 0)

  # determine the best kernel and lambda
  best_idx = np.unravel_index(np.argmax(cv_acc), cv_acc.shape)

  # determine the best kernel and lambda
  best_kernel = kernel_list[best_idx[1]]
  best_lamb = lamb_list[best_idx[0]]

  return cv_acc, best_lamb, best_kernel