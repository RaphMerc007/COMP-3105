# COMP 3105 Assignment 2
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
import numpy as np

import Implementation.A2codes as A2codes
from A2helpers import plotModel, plotAdjModel, plotDualModel, polyKernel, generateData, plotDigit


def _plotCls():
	n = 100
	lamb = 0.01
	gen_model = 1
	kernel_func = lambda X1, X2: polyKernel(X1, X2, 2)

	# Generate data
	Xtrain, ytrain = generateData(n=n, gen_model=gen_model)
	# Learn and plot results
	# Primal
	w, w0 = A2codes.minHinge(Xtrain, ytrain, lamb)
	# w, w0 = A2codes.minExpLinear(Xtrain, ytrain, lamb)
	print("weights", w, " bias ", w0)

	plotModel(Xtrain, ytrain, w, w0, A2codes.classify)

	#get tables for question 1 D and E
	trainAcc, testAcc = A2codes.synExperimentsRegularize()

	print("Regularized Accuracies:")
	print(trainAcc)
	print(testAcc)
	# Adjoint
	a, a0 = A2codes.adjHinge(Xtrain, ytrain, lamb, kernel_func)
	# a, a0 = A2codes.adjExpLinear(Xtrain, ytrain, lamb, kernel_func)

	plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, A2codes.adjClassify)

	trainAcc, testAcc = A2codes.synExperimentsKernel()
	print("Kernel Accuracies:")
	print(trainAcc)
	print(testAcc)

	# Dual
	a, b = A2codes.dualHinge(Xtrain, ytrain, lamb, kernel_func)
	plotDualModel(Xtrain, ytrain, a, b, lamb, kernel_func, A2codes.dualClassify)

	# 1. Create a small test dataset
	# -------------------------
	# X = np.array([
	# 	[1, 2],
	# 	[2, 3],
	# 	[3, 3],
	# 	[2, 1],
	# 	[3, 2]
	# ])
	# y = np.array([1, 1, 1, -1, -1]).reshape(-1,1)  # labels must be ±1

	# lamb = 0.1        # regularization strength
	# stabilizer = 1e-5 # small stabilizer for numerical stability

	# # -------------------------
	# # 2. Call your function
	# # -------------------------
	# # w, w0 = A2codes.minHinge(X, y, lamb, stabilizer)
	# w, w0 = A2codes.minExpLinear(X, y, lamb)


	# # -------------------------
	# # 3. Print results
	# # -------------------------
	# print("Learned weight vector w:")
	# print(w)
	# print("\nLearned bias w0:")
	# print(w0)

	# # Optionally, test on the training data
	# # preds = np.sign(X @ w + w0)
	# preds = A2codes.classify(X, w, w0)
	# print("\nPredictions:", preds)
	# print("Accuracy:", np.mean(preds.flatten() == y.flatten()))

	# #print graph
	# plotModel(X, y, w, w0, A2codes.classify)


def testMnist():
  # FOCUSED TEST - Only best-performing hyperparameters
  print("="*60)
  print("FOCUSED TEST - Top performers only")
  print("="*60)

  # Only test the promising lambda values
  lamb_list = [0.05, 0.1, 0.25, 0.5, 1.0]

  # Only test the best kernels
  kernel_list = []
  kernel_names = []

  # Linear (baseline)
  # kernel_list.append(A2codes.linearKernel)
  # kernel_names.append("Linear")

  # Best Gaussian kernels only (σ = 4-8)
  # for width in [4.0, 5.0, 6.0, 7.0, 8.0]:
  d = 2
  kernel_list.append(lambda X1, X2, d=2: A2codes.polyKernel(X1, X2, d))
  kernel_names.append(f"Poly(d={d})")

  print(f"Testing {len(lamb_list)} lambdas × {len(kernel_list)} kernels = {len(lamb_list) * len(kernel_list)} combinations")
  print(f"Kernels: {kernel_names}")

  # Test with different seed
  cv_acc, best_lamb, best_kernel = A2codes.cvMnist(".", lamb_list, kernel_list, k=5)

  print("\nCV Accuracies:")
  print(cv_acc)

  # Find and display results
  best_idx = np.unravel_index(np.argmax(cv_acc), cv_acc.shape)
  print(f"\nBest Lambda: {best_lamb}")
  print(f"Best Kernel: {kernel_names[best_idx[1]]}")
  print(f"Best Accuracy: {cv_acc[best_idx[0], best_idx[1]]:.4f}")

  # Show top 5 combinations
  flat_acc = cv_acc.flatten()
  top5_indices = np.argsort(flat_acc)[-5:][::-1]
  print("\nTop 5 combinations:")
  for idx in top5_indices:
    lamb_idx, kernel_idx = np.unravel_index(idx, cv_acc.shape)
    print(f"  {kernel_names[kernel_idx]}, λ={lamb_list[lamb_idx]}: {cv_acc[lamb_idx, kernel_idx]:.4f}")	# -------------------------




if __name__ == "__main__":
  # testMnist()
  
	_plotCls()


