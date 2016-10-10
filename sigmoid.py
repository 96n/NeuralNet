# coding: utf-8
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return np.maximum(0,x)

def softmax(x):
	c = np.max(x)
	return np.exp(x-c)/np.sum(np.exp(x-c))
