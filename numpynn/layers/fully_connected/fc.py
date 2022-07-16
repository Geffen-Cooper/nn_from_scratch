''' This file defines the class for fully connected layers'''

import numpy as np
import math
from ..layer import Layer

class FullyConnectedLayer(Layer):
    # input and output neurons define the weight matrix dimensions
    def __init__(self, input_neurons, output_neurons, batch_size, rng):
        self.n_p = input_neurons
        self.n_l = output_neurons

        # initialize the weight matrix with random values
        stdv = 1. / math.sqrt(self.n_p)
        self.W = rng.uniform(-stdv,stdv,(self.n_l,self.n_p))

        # need to store the partial derivatives of the weight matrix
        self.dW = np.zeros((self.n_l,self.n_p))

        # initialize the bias column vector with random values
        self.B = rng.uniform(-stdv,stdv,(self.n_l,1))

        # need to store the partial derivatives of the bias vector
        self.dB = np.zeros((self.n_l,1))

        # need to store the current layer output for backprop
        self.Z = np.zeros((self.n_l,batch_size))

        # need to store previous layer activations for backprop
        self.A_p = np.zeros((self.n_p,batch_size))

    # W*A = Z
    def forward(self, A_p):
        # make sure the dimensions match
        if self.W.shape[1] != A_p.shape[0]:
            print(f'Trying to do W*A_p but W is {W.shape} and A_p is {A_p.shape}')
            raise AssertionError("matrix dimensions invalid")
        
        # since B is a column vector the addition gets casted to the matrix columns
        self.Z = np.dot(self.W,A_p) + self.B

        # save the previous layer activations
        self.A_p = A_p

        return self.Z

    def backward(self, dL_dZ):
        # first find partial derivatives with respect to the weights
        # make sure the dimensions match
        if dL_dZ.shape[1] != self.A_p.T.shape[0]:
            print(f'Trying to do dL_dZ*A_p.T but dL_dZ is {self.W.shape} and A_p.T is {self.A_p.T.shape}')
            raise AssertionError("matrix dimensions invalid")

        self.dW = np.dot(dL_dZ,self.A_p.T) # dL_dW = dL_dZ*dZ_dW

        # next get the partial derivatives with respect to the biases
        self.dB = np.sum(dL_dZ,axis=1,keepdims=True) # dL_dB = dL_dZ

        # now return the local gradient times the upstream gradient
        if self.W.T.shape[1] != dL_dZ.shape[0]:
            print(f'Trying to do W.T*dL_dZ but W.T is {self.W.T.shape} and dL_dZ is {dL_dZ.shape}')
            raise AssertionError("matrix dimensions invalid")
        return np.dot(self.W.T,dL_dZ) # dL_dA = dL_dZ*dZ_dA

    def update_parameters(self, eta, reset=True):
        # subtract parameters by gradient over the batch
        self.W -= eta*self.dW
        self.B -= eta*self.dB

        # reset the gradients
        if reset == True:
            self.dW[:] = 0
            self.dB[:] = 0

    def __str__(self):
        return f'W: shape --> {self.W.shape}\n\n{self.W}\n\nB: shape --> {self.B.shape}\n\n{self.B}'