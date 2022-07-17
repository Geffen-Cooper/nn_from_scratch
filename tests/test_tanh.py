''' This file runs tests on the tanh layer '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from test_fc import create_layer_variables, create_fc_layer
from  numpynn.layers.tanh import tanh as tanh
from  numpynn.layers.fully_connected import fc as fc
import torch
import torch.nn as nn
import numpy as np
from helper_funcs import *


# ===================================================================================== #
# ======================= Helper functions to set up the tests ======================== #
# ===================================================================================== #

# creates a tanh layer in torch and numpynn
def create_tanh_layer(input_dimension):
    torch_tanh = nn.Tanh()
    numpynn_tanh = tanh.TanhLayer(input_dimension)

    return torch_tanh, numpynn_tanh

# creates random layer dimension
def create_random_dimension(dimensions=2):
    # assume 2D inputs for now, TODO: for conv will be higher dimensionality
    input_neurons = rng.integers(1,100)
    batch_size = 2**rng.integers(0,5)
    return (input_neurons, batch_size)


''' ================================== TEST FUNCTIONS ================================='''


# ===================================================================================== #
# ==================================== FORWARD PROP =================================== #
# ===================================================================================== #

# test forward prop
def test_forward():
    # create the layers
    dim = create_random_dimension()
    (torch_tanh,numpynn_tanh) = create_tanh_layer(dim)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch(dim)

    # compare the forward pass outputs
    numpynn_out = numpynn_tanh.forward(numpynn_input)
    torch_out = torch_tanh(torch_input).detach().numpy()

    # implementations slightly different so assert precision threshold of 1e^-5
    out_diff = np.abs(torch_out - numpynn_out)
    assert (out_diff < prec_thresh).all()

# TODO: add a verbose option to the test to see the input, parameters, output
# def print_test():
#     pass


# ===================================================================================== #
# ====================================== BACK PROP ==================================== #
# ===================================================================================== #

# tested by putting a tanh after a fully connected layer
def test_backward():
    # get the fc layer variables
    (input_neurons, output_neurons, batch_size) = create_layer_variables()

    # create the fc layer
    (torch_fc,numpynn_fc) = create_fc_layer(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch((input_neurons, batch_size))

    # create the output labels
    (torch_label, numpynn_label) = create_random_batch((output_neurons, batch_size))

    # create the tanh layers
    (torch_tanh,numpynn_tanh) = create_tanh_layer((output_neurons, batch_size))

    # forward pass
    torch_pred, numpynn_pred = torch_tanh(torch_fc(torch_input.T)), numpynn_tanh.forward(numpynn_fc.forward(numpynn_input))
    
    # Loss = label - output
    # the partial derivative of the Loss w/r/t the outputs is -1 (dL_dy)

    # compute the loss and backprop in pytorch
    torch_loss = torch.sum(torch_label-torch_pred.T)
    torch_loss.backward()

    # manually compute the output gradient and backprop in numpynn
    dL_dy = np.ones(numpynn_label.shape)*-1
    numpynn_fc.backward(numpynn_tanh.backward(dL_dy))

    # compare the calculated gradients for each parameter up to a precision
    diff_dW = np.abs(numpynn_fc.dW - torch_fc.weight.grad.numpy())
    diff_dB = np.abs(numpynn_fc.dB - torch_fc.bias.grad.numpy().reshape(numpynn_fc.dB.shape))
    
    assert ((diff_dW) < prec_thresh).all()
    assert ((diff_dB) < prec_thresh).all()