#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class Neuron:
    '''
        Private instance attributes:
            W: The weights vector for the neuron. Upon instantiation,
               it should be initialized using a random normal distribution.
            b: The bias for the neuron. Upon instantiation,
               it should be initialized to 0.
            A: The activated output of the neuron (prediction).
               Upon instantiation, it should be initialized to 0.
    '''
    def __init__(self, nx):
        '''
            class constructor: def __init__(self, nx):
                nx is the number of input features to the neuron
            If nx is not an integer,
                raise a TypeError with the exception: nx must be an integer
            If nx is less than 1,
                raise a ValueError with the exception:
                nx must be a positive integer
            All exceptions should be raised in the order listed above
        '''
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
            Getter function
            Returns:
                numpy.ndarray
                The weight vector of the neuron.
        '''
        return self.__W

    @property
    def b(self):
        '''
            Getter function
            Returns:
                float
                The bias term of the neuron.
        '''
        return self.__b

    @property
    def A(self):
        '''
            Getter function
             Returns:
            float
                The activated output of the neuron.
        '''
        return self.__A

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of the neuron
            X is a numpy.ndarray with shape(nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Updates the private attribute __A
            The neuron should use a sigmoid activation function
            Returns the private attribute __A
        '''
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1.0 / (1.0 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''
            Calculates the cost of the model using logistic regression

            Y is a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data

            A is a numpy.ndarray with shape (1, m)
            containing the activated output of the neuron for each example

            To avoid division by zero errors,
            please use 1.0000001 - A instead of 1 - A

            Returns the cost
        '''
        # logistic regression cost function
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        '''
            Evaluates the neuron's predictions
            X is a numpy.ndarray with shape(nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y is a numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data
            Returns the neuron's prediction and the cost of the network,
            respectively
                The prediction should be a numpy.ndarray with shape (1, m)
                    containing the predicted labels for each example
                The label values should be 1 if the output of the
                    network is >= 0.5 and 0 otherwise
        '''
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
            Calculates one pass of gradient descent on the neuron
            X is a numpy.ndarray with shape(nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y is a numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data
            A is a numpy.ndarray with shape (1, m)
                containing the activated output of the neuron for each example
            alpha is the learning rate
            Updates the private attributes __W and __b
        '''
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - alpha * db
