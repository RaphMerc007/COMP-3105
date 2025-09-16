# COMP 3105 Fall 2025 Assignment 1
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions). 
#       We will use a different script to test your codes. 
import numpy as np
from matplotlib import pyplot as plt
import Implementation.A1codes as A1codes


def _plotReg():

	# simple 2D example
	n = 30  # number of data points
	d = 1  # dimension
	noise = 0.2  # noise level
	X = np.random.randn(n, d)  # input matrix
	y = X + np.random.randn(n, 1) * noise + 2  # ground truth label

	plt.scatter(X, y, marker='x', color='k')  # plot data points

	# learning
	X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment input
	w_L2 = A1codes.minimizeL2(X, y)
	y_hat_L2 = X @ w_L2
	w_Linf = A1codes.minimizeLinf(X, y)
	y_hat_Linf = X @ w_Linf

	# plot models
	plt.plot(X[:, 1], y_hat_L2, 'b', marker=None, label='$L_2$')
	plt.plot(X[:, 1], y_hat_Linf, 'r', marker=None, label='$L_\infty$')
	plt.legend()
	plt.show()


def _plotCls():

	# 2D classification example
	m = 100
	d = 2
	c0 = np.array([[1, 1]])  # cls 0 center
	c1 = np.array([[-1, -1]])  # cls 1 center
	X0 = np.random.randn(m, 2) + c0
	X1 = np.random.randn(m, 2) + c1

	# plot data points
	plt.scatter(X0[:, 0], X0[:, 1], marker='x', label='Negative')
	plt.scatter(X1[:, 0], X1[:, 1], marker='o', label='Positive')

	X = np.concatenate((X0, X1), axis=0)
	X = np.concatenate((np.ones((2*m, 1)), X),  axis=1)  # augment input
	y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)  # class labels

	# find optimal solution
	w_opt = A1codes.find_opt(A1codes.logisticRegObj, A1codes.logisticRegGrad, X, y)

	# plot models
	x_grid = np.arange(-4, 4, 0.01)
	plt.plot(x_grid, (-w_opt[0]-w_opt[1]*x_grid) / w_opt[2], '--k')
	plt.legend()
	plt.show()


if __name__ == "__main__":

	_plotReg()
	_plotCls()
