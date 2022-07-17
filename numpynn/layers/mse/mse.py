''' This file defines the class for MSE (mean squared error) loss layer'''

import numpy as np
import math
from ..layer import Layer

class MSELayer(Layer):
    # need the dimension of the input activations
    def __init__(self, input_dimension):
        self.dim = input_dimension

        # store output value for backprop
        self.Y = np.zeros(self.dim) 
        self.Y_hat = np.zeros(self.dim)

    # L(Y,Y_hat)
    def forward(self, Y_hat, Y):
        # make sure the dimensions match
        if (Y_hat.shape != self.Y.shape) or (Y.shape != self.Y.shape):
            print(f'Expected shape is {self.Y.shape} and Y,Y_hat are {Y.shape},{Y_hat.shape}')
            raise AssertionError("matrix dimensions invalid")
        
        self.Y = Y
        self.Y_hat = Y_hat
        return 1/(Y.shape[0]*Y.shape[1])*np.sum(np.square(Y-Y_hat))

    # this is the most upstream gradient dL_dY
    def backward(self):
        return 2/(self.Y.shape[0]*self.Y.shape[1])*(self.Y_hat-self.Y)

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'A: shape --> {self.A.shape}\n'