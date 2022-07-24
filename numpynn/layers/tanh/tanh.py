''' This file defines the class for tanh activation layers'''

import numpy as np
import math
from ..layer import Layer

class TanhLayer(Layer):
    # g(Z)
    def forward(self, Z):
        x1 = np.exp(Z)
        x2 = np.exp(-Z)
        # store activation for backpropr
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