''' This file runs tests on the network class '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from numpynn.layers.relu import relu as relu
from  numpynn.layers.mse import mse as mse
from  numpynn.layers.fully_connected import fc as fc
from numpynn import Network as net
import torch
import torch.nn as nn
import numpy as np
from helper_funcs import *

# first define the pytorch model
class TorchFCNet(nn.Module):
    def __init__(self, input_neurons, output_neurons, num_layers):
        super(TorchFCNet, self).__init__()
        # list of layers
        self.layers = nn.ModuleList()

        # add multiple layers with random hidden neurons
        for layer in range(num_layers):
            hidden_neurons = rng.integers(3,10)
            self.layers.append(nn.Linear(input_neurons, hidden_neurons))
            input_neurons = hidden_neurons

        # final layer
        self.layers.append(nn.Linear(input_neurons, output_neurons))

        # activation
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x  

# copy the pytorch weights and architecture into the numpy model
def create_fc_nets():
    # create the pytorch model
    torch_fc_net = TorchFCNet(rng.integers(2,10),rng.integers(2,10),rng.integers(2,5))
    
    # create the numpynn model
    numpynn_fc_net = net.Network()

    # create the numpynn layers and copy the initialized parameters from pytorch
    for torch_layer in torch_fc_net.layers:
        input_neurons,output_neurons = torch_layer.weight.size()[1], torch_layer.weight.size()[0]
        numpynn_layer = fc.FullyConnectedLayer(input_neurons,output_neurons,rng)
        numpynn_layer.W = torch_layer.weight.detach().numpy()
        numpynn_layer.B = torch_layer.bias.detach().numpy().reshape(numpynn_layer.B.shape)
        numpynn_fc_net.add_layer(numpynn_layer)
        activation = relu.ReLULayer()
        numpynn_fc_net.add_layer(activation)

    return torch_fc_net, numpynn_fc_net


def test_fc_network_forward():
    torch_fc_net, numpynn_fc_net = create_fc_nets()

    feature_d = numpynn_fc_net.layers[0].W.shape[1]
    batch_size = 2**rng.integers(0,5)
    torch_batch, numpynn_batch = create_random_batch((feature_d,batch_size))

    torch_out = torch_fc_net(torch_batch.T).detach().numpy().T
    numpynn_out = numpynn_fc_net.forward_pass(numpynn_batch)

    # implementations slightly different so assert precision threshold of 1e^-5
    out_diff = np.abs(torch_out - numpynn_out)
    assert (out_diff < prec_thresh).all()