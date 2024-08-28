#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class NeuralNetwork:
    '''
        class constructor: def __init__(self, nx, nodes):
            nx is the number of input features
                If nx is not an integer, raise a TypeError
                    with the exception: nx must be an integer
                If nx is less than 1, raise a ValueError
                    with the exception: nx must be a positive integer
            nodes is the number of nodes found in the hidden layer
                If nodes is not an integer, raise a TypeError
                    with the exception: nodes must be an integer
                If nodes is less than 1, raise a ValueError
                    with the exception: nodes must be a positive integer
            All exceptions should be raised in the order listed above
        Public instance attributes:
            W1: The weights vector for the hidden layer. Upon instantiation,
                it should be initialized using a random normal distribution.
            b1: The bias for the hidden layer. Upon instantiation,
                it should be initialized with 0's.
            A1: The activated output for the hidden layer.
                Upon instantiation, it should be initialized to 0.
            W2: The weights vector for the output neuron. Upon instantiation,
                it should be initialized using a random normal distribution.
            b2: The bias for the output neuron. Upon instantiation,
                it should be initialized to 0.
            A2: The activated output for the output neuron (prediction).
                Upon instantiation, it should be initialized to 0.
    '''
    if isinstance(nx, int) is False:
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")
    if isinstance(nodes, int) is False:
        raise TypeError("nodes must be an integer")
    if nodes < 1:
        raise ValueError("nodes must be a positive integer")
    self.W1 = np.random.normal(0, 1, (nodes, nx))
    self.b1 = np.zeros((nodes, 1))
    self.A1 = 0
    self.W2 = np.random.normal(0, 1, (1, nodes))
    self.b2 = 0
    self.A2 = 0
