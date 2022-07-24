''' This file defines the class for sigmoid activation layers'''

import numpy as np
import math
from ..layer import Layer

class SigmoidLayer(Layer):
    # g(Z)
    def forward(self, Z):
        # store activation for backpropr
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    # multiply elementwise the derivative of the activation function by the upstream gradient and return
    def backward(self, dL_dA):
        # activation represents sig(z) and sig_prime(z) = sig(z)*(1-sig(z))
        dA_dZ = np.multiply(self.A,(1-self.A))
        return np.multiply(dA_dZ, dL_dA) # dL_dZ = dL_dA*dA_dZ

    def update_parameters(self, eta, reset=True):
        # has no parameters
        pass

    def __str__(self):
        return f'A: shape --> {self.A.shape}\n'