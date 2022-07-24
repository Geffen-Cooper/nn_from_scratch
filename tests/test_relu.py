''' This file runs tests on the relu layer '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from  numpynn.layers.relu import relu as relu
from  numpynn.layers.fully_connected import fc as fc
import torch
import torch.nn as nn
import numpy as np
from helper_funcs import *


# ===================================================================================== #
# ======================= Helper functions to set up the tests ======================== #
# ===================================================================================== #

# creates a sigmoid layer in torch and numpynn
def create_relu_layer():
    torch_relu = nn.ReLU()
    numpynn_relu = relu.ReLULayer()

    return torch_relu, numpynn_relu


''' ================================== TEST FUNCTIONS ================================='''


# ===================================================================================== #
# ==================================== FORWARD PROP =================================== #
# ===================================================================================== #

# test forward prop
def test_forward():
    # create the layers
    (feature_d, batch_size) = create_random_dimension()
    (torch_relu,numpynn_relu) = create_relu_layer()

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch((feature_d, batch_size))

    # compare the forward pass outputs
    numpynn_out = numpynn_relu.forward(numpynn_input)
    torch_out = torch_relu(torch_input).detach().numpy()

    # implementations slightly different so assert precision threshold of 1e^-5
    out_diff = np.abs(torch_out - numpynn_out)
    assert (out_diff < prec_thresh).all()

# TODO: add a verbose option to the test to see the input, parameters, output
# def print_test():
#     pass


# ===================================================================================== #
# ====================================== BACK PROP ==================================== #
# ===================================================================================== #

# tested by putting a sigmoid after a fully connected layer
def test_backward():
    # get the fc layer variables
    (input_neurons, output_neurons, batch_size) = create_fc_layer_variables()

    # create the fc layer
    (torch_fc,numpynn_fc) = create_fc_layer(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch((input_neurons, batch_size))

    # create the output labels
    (torch_label, numpynn_label) = create_random_batch((output_neurons, batch_size))

    # create the sigmoid layers
    (torch_relu,numpynn_relu) = create_relu_layer()

    # forward pass
    torch_pred, numpynn_pred = torch_relu(torch_fc(torch_input.T)), numpynn_relu.forward(numpynn_fc.forward(numpynn_input))
    
    # Loss = label - output
    # the partial derivative of the Loss w/r/t the outputs is -1 (dL_dy)

    # compute the loss and backprop in pytorch
    torch_loss = torch.sum(torch_label-torch_pred.T)
    torch_loss.backward()

    # manually compute the output gradient and backprop in numpynn
    dL_dy = np.ones(numpynn_label.shape)*-1
    numpynn_fc.backward(numpynn_relu.backward(dL_dy))

    # compare the calculated gradients for each parameter up to a precision
    diff_dW = np.abs(numpynn_fc.dW - torch_fc.weight.grad.numpy())
    diff_dB = np.abs(numpynn_fc.dB - torch_fc.bias.grad.numpy().reshape(numpynn_fc.dB.shape))
    
    assert ((diff_dW) < prec_thresh).all()
    assert ((diff_dB) < prec_thresh).all()