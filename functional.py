import numpy as np

def softmax(X):
	n = X.shape[0]
	exps = np.exp(X - np.max(X, axis = 1).reshape(n, 1))
	return exps / np.sum(exps, axis = 1).reshape(n, 1)

def softmax_grad(X, y):
	return X - y

def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-X))

def sigmoid_grad(X):
	return sigmoid(X) * (1 - sigmoid(X))

def relu(X):
	return (X > 0) * X

def relu_grad(X):
	return (X > 0) * 1

def glorot(input_size, output_size):
	sqrt = np.sqrt(1.0 / output_size)
	return np.random.uniform(sqrt * -1, sqrt, [input_size, output_size])