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
