''' This file defines the class for the cross entropy loss layer'''

import numpy as np
import math
from ..layer import Layer

class CrossEntropyLayer(Layer):
    # L(Y,Y_hat)
    def forward(self, Y_hat, Y):
        # make sure the dimensions match
        if (Y_hat.shape[1] != Y.shape[1]):
            print(f'Expected shape is {Y.shape} and Y,Y_hat are {Y.shape},{Y_hat.shape}')
            raise AssertionError("matrix dimensions invalid")
        
        # store output and expected output for backprop
        self.Y = Y
        self.Y_hat = Y_hat

        # Here we apply cross entropy to the columns (each sample) then sum
        # We assume Y is a row vector with each entry being a label for each sample

        # the first index is the sample (col), the second index is the label (row)
        return (1/Y.shape[1])*-np.sum(np.log(Y_hat[Y,np.arange(Y.shape[1])]))

    # this is the most upstream gradient dL_dY
    def backward(self):
        one_hot_label = np.zeros(self.Y_hat.shape)
        one_hot_label[self.Y,np.arange(self.Y.shape[1])] = 1
        self.Y_hat -= one_hot_label
        self.Y_hat *= (1/self.Y.shape[1])
        return self.Y_hat

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'Y_hat: shape --> {self.Y_hat.shape}\n'