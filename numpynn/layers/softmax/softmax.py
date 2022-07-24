''' This file defines the class for the softmax activation layer'''

import numpy as np
import math
from ..layer import Layer

class SoftmaxLayer(Layer):
    # g(Z)
    def forward(self, Z):
        # We should apply the softmax for each column since the output of each sample
        # is a column vector. This is transpose of the pytorch convention
        self.A = np.exp(Z) / np.sum(np.exp(Z),axis=0)
        return self.A

    # multiply elementwise the derivative of the activation function by the upstream gradient and return
    def backward(self, dL_dA):
        # for now just return the upstream gradient since the cross entropy
        # will calculate dL_dA*dA_dZ so we don't need to find dA_dZ here
        return dL_dA

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'A: shape --> {self.A.shape}\n'