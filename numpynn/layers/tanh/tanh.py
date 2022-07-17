''' This file defines the class for tanh activation layers'''

import numpy as np
import math
from ..layer import Layer

class TanhLayer(Layer):
    # need the dimension of the input activations
    def __init__(self, input_dimension):
        self.dim = input_dimension

        # store activations for backprop
        self.A = np.zeros(self.dim) 

    # g(Z)
    def forward(self, Z):
        # make sure the dimensions match
        if Z.shape != self.A.shape:
            print(f'Expected shape is {self.A.shape} and Z is {Z.shape}')
            raise AssertionError("matrix dimensions invalid")
        
        x1 = np.exp(Z)
        x2 = np.exp(-Z)
        self.A = (x1 - x2) / (x1 + x2)
        return self.A

    # multiply elementwise the derivative of the activation function by the upstream gradient and return
    def backward(self, dL_dA):
        # activation represents tanh(z) and tanh_prime(z) = 1 - (tanh(z))^2
        dA_dZ = 1 - np.square(self.A)
        return np.multiply(dA_dZ, dL_dA) # dL_dZ = dL_dA*dA_dZ

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'A: shape --> {self.A.shape}\n'