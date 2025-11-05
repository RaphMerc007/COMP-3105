# COMP 3105 Assignment 3
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
from matplotlib import pyplot as plt

import Implementation.A3codes as A3codes
from A3helpers import augmentX, gaussKernel, plotModel, generateData, plotPoints


def _plotCls():

	n = 100

	# Generate data
	Xtrain, Ytrain = generateData(n=n, gen_model=1, rand_seed=0)
	Xtrain = augmentX(Xtrain)

	# Learn and plot results
	W = A3codes.minMulDev(Xtrain, Ytrain)
	print(f"Train accuaracy {A3codes.calculateAcc(Ytrain, A3codes.classify(Xtrain, W))}")

	plotModel(Xtrain, Ytrain, W, A3codes.classify)

	return


def _testPCA():
	train_acc, test_acc = A3codes.synClsExperimentsPCA()
	print(train_acc)
	print(test_acc)
	return


def _plotKmeans():

	n = 100
	k = 3

	Xtrain, _ = generateData(n, gen_model=2)

	Y, U, obj_val = A3codes.kmeans(Xtrain, k)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()

	return


def _plotKernelKmeans():
	Xtrain, _ = generateData(n=100, gen_model=3)
	kernel_func = lambda X1, X2: gaussKernel(X1, X2, 0.25)
	init_Y = None  # TODO: you need to change this
	Y, obj_val = A3codes.kernelKmeans(Xtrain, kernel_func, 2, init_Y)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()
	return


if __name__ == "__main__":

	_plotCls()
	# _testPCA()
	# _plotKmeans()
	# _plotKernelKmeans()
