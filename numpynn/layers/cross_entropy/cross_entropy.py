''' This file defines the class for the cross entropy loss layer'''

import numpy as np
import math
from ..layer import Layer

class CrossEntropyLayer(Layer):
    # need the dimension of the input activations
    def __init__(self, input_dimension):
        self.dim = input_dimension

        # store output value for backprop
        self.Y = np.zeros((1,self.dim[1])) 
        self.Y_hat = np.zeros(self.dim)

    # L(Y,Y_hat)
    def forward(self, Y_hat, Y):
        # make sure the dimensions match
        if (Y_hat.shape[1] != self.Y.shape[1]):
            print(f'Expected shape is {self.Y.shape} and Y,Y_hat are {Y.shape},{Y_hat.shape}')
            raise AssertionError("matrix dimensions invalid")
        
        self.Y = Y
        self.Y_hat = Y_hat

        # Here we apply cross entropy to the columns (each sample) then sum
        # We assume Y is a row vector with each entry being a label for each sample

        # the first index is the sample (col), the second index is the label (row)
        return (1/Y.shape[1])*-np.sum(np.log(Y_hat[Y,np.arange(Y.shape[1])]))

    # this is the most upstream gradient dL_dY
    def backward(self):
        return self.Y_hat-self.Y

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'A: shape --> {self.A.shape}\n'