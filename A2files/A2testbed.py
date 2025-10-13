# COMP 3105 Assignment 2
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
import numpy as np

import Implementation.A2codes as A2codes
from A2helpers import plotModel, plotAdjModel, plotDualModel, polyKernel, generateData


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
	plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, A2codes.adjClassify)

	trainAcc, testAcc = A2codes.synExperimentsKernel()
	print("Kernel Accuracies:")
	print(trainAcc)
	print(testAcc)

	# # Dual
	# a, b = A2codes.dualHinge(Xtrain, ytrain, lamb, kernel_func)
	# plotDualModel(Xtrain, ytrain, a, b, lamb, kernel_func, A2codes.dualClassify)

	# #TODO: (TINY TEST DATASET) remove after done testing q1 purposes below and uncomment lines above
	# -------------------------
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






if __name__ == "__main__":

	_plotCls()
