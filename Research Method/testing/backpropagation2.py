#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:01:15 2021

@author: husnids
"""
import numpy as np
# passed gradient from the next layer
dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])

# we have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                   [0.5, -0.91, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]]).T

inputs = np.array([[1, 2, 3, 2.5],
                   [2, 5, -1, 2],
                   [-1.5, 2.7, 3.3, -0.8]])                    
# one bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# sum weights of given output
dx0 = sum(weights[0])*dvalues[0] 
dx1 = sum(weights[1])*dvalues[0]
dx2 = sum(weights[2])*dvalues[0]
dx3 = sum(weights[3])*dvalues[0]

dinputs = np.dot(dvalues, weights.T) #gradient of the neuron function with respect to inputs
#dinputs = np.array([dx0, dx1, dx2, dx3]) 
print(dinputs)

dweights = np.dot(inputs.T, dvalues) #gradient of the neuron function with respect to weights
print(dweights)

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases) # keepdims -> let us to keep the gradient as a row vector-recall the shape of biases array