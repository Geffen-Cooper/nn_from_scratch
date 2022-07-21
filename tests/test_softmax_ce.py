''' This file runs tests on the softmax cross entropy combo layer '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from test_fc import create_layer_variables, create_fc_layer
from  numpynn.layers.softmax import softmax as sm
from  numpynn.layers.cross_entropy import cross_entropy as ce
from  numpynn.layers.fully_connected import fc as fc
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from helper_funcs import *


# ===================================================================================== #
# ======================= Helper functions to set up the tests ======================== #
# ===================================================================================== #

# creates a softmax-cross entropy layer in torch and numpynn
def create_smce_layer(input_dimension):
    torch_smce = nn.CrossEntropyLoss()
    numpynn_sm = sm.SoftmaxLayer(input_dimension)
    numpynn_ce = ce.CrossEntropyLayer(input_dimension)

    return torch_smce, numpynn_sm, numpynn_ce

# creates random layer dimension
def create_random_dimension(dimensions=2):
    # assume 2D inputs for now, TODO: for conv will be higher dimensionality
    input_neurons = rng.integers(1,100)
    batch_size = 2**rng.integers(0,5)
    print(input_neurons,batch_size)
    return (input_neurons, batch_size)


''' ================================== TEST FUNCTIONS ================================='''


# ===================================================================================== #
# ==================================== FORWARD PROP =================================== #
# ===================================================================================== #

# test forward prop
def test_forward():
    # create the layers
    dim = create_random_dimension()
    (torch_smce, numpynn_sm, numpynn_ce) = create_smce_layer(dim)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch(dim)

    # create the output labels (classification so column vector)
    (torch_label, numpynn_label) = create_random_batch((1,dim[1]))
    torch_label = (torch_label[0]*3).long()
    numpynn_label = (numpynn_label*3).astype(int)

    # compare the forward pass outputs
    # torch expects a (N,) dim label vector and the softmax dim must be specified
    
    numpynn_out = numpynn_ce.forward(numpynn_sm.forward(numpynn_input), numpynn_label)
    torch_out = torch_smce(torch_input.T, torch_label).detach().numpy()

    # implementations slightly different so assert precision threshold of 1e^-5
    out_diff = np.abs(torch_out - numpynn_out)
    assert (out_diff < prec_thresh).all()

# TODO: add a verbose option to the test to see the input, parameters, output
# def print_test():
#     pass


# ===================================================================================== #
# ====================================== BACK PROP ==================================== #
# ===================================================================================== #

# # tested by putting a MSE after a fully connected layer
# def test_backward():
#     # get the fc layer variables
#     (input_neurons, output_neurons, batch_size) = create_layer_variables()

#     # create the fc layer
#     (torch_fc,numpynn_fc) = create_fc_layer(input_neurons,output_neurons)

#     # create the input batch
#     (torch_input, numpynn_input) = create_random_batch((input_neurons, batch_size))

#     # create the output labels
#     (torch_label, numpynn_label) = create_random_batch((output_neurons, batch_size))

#     # create the MSE layers
#     (torch_mse,numpynn_mse) = create_mse_layer((output_neurons, batch_size))

#     # forward pass
#     torch_pred, numpynn_pred = torch_fc(torch_input.T), numpynn_fc.forward(numpynn_input)
    
#     # Loss = MSE(label, output)

#     # compute the loss and backprop in pytorch
#     torch_loss = torch_mse(torch_pred.T,torch_label)
#     torch_loss.backward()

#     # manually compute the output gradient and backprop in numpynn
#     numpynn_loss = numpynn_mse.forward(numpynn_pred,numpynn_label)
#     numpynn_fc.backward(numpynn_mse.backward())

#     # compare the calculated gradients for each parameter up to a precision
#     diff_dW = np.abs(numpynn_fc.dW - torch_fc.weight.grad.numpy())
#     diff_dB = np.abs(numpynn_fc.dB - torch_fc.bias.grad.numpy().reshape(numpynn_fc.dB.shape))
    
#     assert ((diff_dW) < prec_thresh).all()
#     assert ((diff_dB) < prec_thresh).all()