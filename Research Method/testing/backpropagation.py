#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:09:59 2021
== Backpropagation ==
@author: husnids
"""

# forward pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weight
b = 1.0 # bias

# multiplying inputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

# adding weighted inputs and a bias 
z = xw0 +xw1 + xw2 + b

# Relu activation function
y = max(z, 0)

# ===== backward pass =====

# the derivative from the next layer 
dvalue = 1.0

# derivative of ReLU and the chain rule 
drelu_dz = dvalue*(1. if z > 0 else 0.)
#print(drelu_dz)

# partial derivative of the of the sum, the chain rule
dsum_dxw0 = 1 
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

drelu_dxw0 = dsum_dxw0*drelu_dz
drelu_dxw1 = dsum_dxw1*drelu_dz
drelu_dxw2 = dsum_dxw2*drelu_dz
drelu_db = drelu_dz*drelu_dz
#print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# partial derivatives of the multiplication, the chain rule
# with respect to x
dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0*dmul_dx0
dmul_dx1 = w[1]
drelu_dx1 = drelu_dxw1*dmul_dx1
dmul_dx2 = w[2]
drelu_dx2 = drelu_dxw2*dmul_dx2
# with respect to w
dmul_dw0 = x[0]
drelu_dw0 = drelu_dxw0*dmul_dw0
dmul_dw1 = x[1]
drelu_dw1 = drelu_dxw1*dmul_dw1
dmul_dw2 = x[2]
drelu_dw2 = drelu_dxw2*dmul_dw2

#print(drelu_dx0,drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# gradients
dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradient on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradient on weights
db = drelu_db # gradient one bias ... just 1 bias here

# === apply a fraction of the gradients to weights: ===
w[0] += -0.001*dw[0]
w[1] += -0.001*dw[1]
w[2] += -0.001*dw[2]
b += -0.001*db

# multiplying inputs by weights so we can get weighted
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]
# adding
z = xw0+xw1+xw2+b
# ReLU activation func
y = max(z,0)
print(y)














