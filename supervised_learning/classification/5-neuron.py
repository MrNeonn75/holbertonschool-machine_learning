#!/usr/bin/env python3
"""neuron class"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """init function"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation function"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """cost function"""
        m = Y.shape[1]
        return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """evaluate function"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.round(A).astype(int), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent function"""
        m = Y.shape[1]
        dz = A - Y
        dw = 1 / m * np.dot(X, dz.T)
        db = 1 / m * np.sum(dz)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
        return self.__W, self.__b
