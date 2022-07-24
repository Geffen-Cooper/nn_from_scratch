''' This file runs tests on the fully connected layer '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from  numpynn.layers.fully_connected import fc as fc
import torch
import torch.nn as nn
import numpy as np
from helper_funcs import *


''' ================================== TEST FUNCTIONS ================================='''


# ===================================================================================== #
# ==================================== FORWARD PROP =================================== #
# ===================================================================================== #

def test_forward():
    # get the layer variables
    (input_neurons, output_neurons, batch_size) = create_fc_layer_variables()

    # create the layers
    (torch_fc,numpynn_fc) = create_fc_layer(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch((input_neurons, batch_size))

    # compare the forward pass outputs, note that torch does xW^T and numpynn does Wx
    numpynn_out = numpynn_fc.forward(numpynn_input)
    torch_out = torch_fc(torch_input.T).detach().numpy().T

    # implementations slightly different so assert precision threshold of 1e^-5
    out_diff = np.abs(torch_out - numpynn_out)
    assert (out_diff < prec_thresh).all()

# TODO: add a verbose option to the test to see the input, parameters, output
# def print_test():
#     print("=== pytorch fc:===\n",torch_fc.weight,torch_fc.bias,torch_input)
#     print("\n===numpynn fc:===\n",numpynn_fc,numpynn_input)



# ===================================================================================== #
# ===================================== BACK PROP ===================================== #
# ===================================================================================== #

def test_backward():
    # get the layer variables
    (input_neurons, output_neurons, batch_size) = create_fc_layer_variables()

    # create the layers
    (torch_fc,numpynn_fc) = create_fc_layer(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_random_batch((input_neurons, batch_size))

    # create the output labels
    (torch_label, numpynn_label) = create_random_batch((output_neurons, batch_size))

    # get the output
    torch_pred, numpynn_pred = torch_fc(torch_input.T), numpynn_fc.forward(numpynn_input)

    # Loss = label - output
    # the partial derivative of the Loss w/r/t the outputs is -1 (dL_dy)

    # compute the loss and backprop in pytorch
    torch_loss = torch.sum(torch_label-torch_pred.T)
    torch_loss.backward()

    # manually compute the output gradient and backprop in numpynn
    dL_dy = np.ones((output_neurons,batch_size))*-1
    numpynn_fc.backward(dL_dy)

    # compare the calculated gradients for each parameter up to a precision
    diff_dW = np.abs(numpynn_fc.dW - torch_fc.weight.grad.numpy())
    diff_dB = np.abs(numpynn_fc.dB - torch_fc.bias.grad.numpy().reshape(numpynn_fc.dB.shape))
    
    assert ((diff_dW) < prec_thresh).all()
    assert ((diff_dB) < prec_thresh).all()