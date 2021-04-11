#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:41:50 2021
@author: husnids
Combined forward and backward pass of a single neuron with full layer 
and batch based partial derivatives.
"""

import numpy as np
# passed in gradient from the next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]]) 

# we have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# we have 3 sets of weights - one set for each neuron
# 4 inputs thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# one bias for each neuron 
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]]) 

# ========== FORWARD PASS =========
layer_outputs = np.dot(inputs, weights)+biases # Dense Layer 
relu_outputs = np.maximum(0, layer_outputs) # ReLU activation 

# ========== BACKPROPAGATION =========
# ReLU activation's derivative
# from next layer passed the current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer 
# dinputs: gradient of the neuron function with respect to inputs
dinputs = np.dot(drelu, weights.T)
# dweights: gradient of neuron function with respect to weights
dweights = np.dot(inputs.T, drelu)
# dbiases
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights #update weights
biases += -0.001 * dbiases # update biases

print(weights)
print(biases)