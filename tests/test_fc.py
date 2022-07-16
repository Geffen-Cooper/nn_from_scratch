''' This file runs tests on the fully connected layer '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from  numpynn.layers.fully_connected import fc as fc
import torch
import torch.nn as nn
import numpy as np


# ======================== Global variables used for the test ========================
rng = np.random.default_rng(seed=42)
prec_thresh = 10**(-5)

# ======================= Helper functions to set up the tests =======================

# creates a fully connected layer in torch and numpynn with the same initial parameters
def create_fc_layers(input_neurons, output_neurons):
    # let pytorch determine the parameters randomly
    torch_fc = nn.Linear(input_neurons,output_neurons)
    numpynn_fc = fc.FullyConnectedLayer(input_neurons,output_neurons,1,rng)

    # copy the values to numpynn
    numpynn_fc.W = torch_fc.weight.detach().numpy()
    numpynn_fc.B = torch_fc.bias.detach().numpy().reshape(numpynn_fc.B.shape)

    return torch_fc,numpynn_fc

# creates a random input vector to test the layer
def create_input_batch(input_features, batch_size):
    # let pytorch determine input and copy to numpynn
    torch_input = torch.rand((input_features,batch_size))
    numpynn_input = torch_input.detach().numpy()

    return torch_input, numpynn_input

# create random output labels for the input batch
def create_output_labels(output_features,batch_size):
    # let pytorch determine output and copy to numpynn
    torch_label = torch.rand((output_features,batch_size))
    numpynn_label = torch_label.detach().numpy()

    return torch_label, numpynn_label

# creates random layer variables
def create_layer_variables():
    input_neurons = rng.integers(1,100)
    output_neurons = rng.integers(1,10)
    batch_size = 2**rng.integers(0,5)
    return input_neurons, output_neurons, batch_size


''' ======================== TEST FUNCTIONS ========================'''
# test forward prop
def test_forward():
    # get the layer variables
    (input_neurons, output_neurons, batch_size) = create_layer_variables()

    # create the layers
    (torch_fc,numpynn_fc) = create_fc_layers(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_input_batch(input_neurons, batch_size)

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

# test backprop
def test_backward():
    # get the layer variables
    (input_neurons, output_neurons, batch_size) = create_layer_variables()

    # create the layers
    (torch_fc,numpynn_fc) = create_fc_layers(input_neurons,output_neurons)

    # create the input batch
    (torch_input, numpynn_input) = create_input_batch(input_neurons, batch_size)

    # create the output labels
    (torch_label, numpynn_label) = create_output_labels(output_neurons, batch_size)

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
    diff_dB = np.abs(numpynn_fc.dB - torch_fc.bias.grad.numpy())
    
    assert ((diff_dW) < prec_thresh).all()
    assert ((diff_dB) < prec_thresh).all()