''' This file runs tests on full training of a fc net '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
from numpynn.layers.relu import relu as relu
from  numpynn.layers.cross_entropy import cross_entropy as ce
from numpynn.layers.softmax import softmax as sm
from  numpynn.layers.fully_connected import fc as fc
from numpynn import Network as net
from numpynn.data.generate_data import *
import torch
import torch.nn as nn
import numpy as np
from helper_funcs import *

# first define the pytorch model
class TorchFCNet(nn.Module):
    def __init__(self,input_neurons, output_neurons):
        super(TorchFCNet, self).__init__()
        # list of layers
        self.layers = nn.ModuleList()

        self.fc1 = nn.Linear(input_neurons,5)
        self.fc2 = nn.Linear(5,output_neurons)
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  

# copy the pytorch weights and architecture into the numpy model
def create_fc_nets(input_neurons, output_neurons):
    # create the pytorch model
    torch_fc_net = TorchFCNet(input_neurons, output_neurons)
    
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
    numpynn_fc_net.layers.pop(-1) # don't want the last relu

    return torch_fc_net, numpynn_fc_net


def train_fc_classifiers(epochs,lr,batch_size):
    # create the models
    torch_fc_net, numpynn_fc_net = create_fc_nets(2,3)

    # create the data
    X_train,Y_train,X_test,Y_test = gen_clusters(test_split=0.2)
    num_batches = X_train.shape[0] // batch_size

    # create the loss functions
    torch_smce = nn.CrossEntropyLoss()
    numpynn_sm = sm.SoftmaxLayer()
    numpynn_ce = ce.CrossEntropyLayer()

    # torch optimizer
    torch_optimizer = torch.optim.SGD(torch_fc_net.parameters(),lr=lr)

    torch_losses = np.zeros(num_batches)
    numpynn_losses = np.zeros(num_batches)

    # training loop
    for epoch in range(epochs):
        # shuffle the training data
        rand_idxs = np.random.permutation(len(X_train))
        for batch_idx in range(num_batches):
            torch_optimizer.zero_grad()

            # get a random batch from the training data
            idxs = rand_idxs[batch_idx*batch_size:batch_idx*batch_size+batch_size]
            batch_X, batch_Y = X_train[idxs],Y_train[idxs]

            # forward pass
            torch_out = torch_fc_net(torch.from_numpy(batch_X).float())
            numpynn_out = numpynn_fc_net.forward_pass(batch_X.T)

            # loss calculation
            torch_loss = torch_smce(torch_out,torch.from_numpy(batch_Y).long())
            numpynn_loss = numpynn_ce.forward(numpynn_sm.forward(numpynn_out),np.expand_dims(batch_Y,axis=0))

            torch_losses[batch_idx] = torch_loss.detach().item()
            numpynn_losses[batch_idx] = numpynn_loss

            # backward pass
            torch_loss.backward()
            numpynn_fc_net.backward_pass(numpynn_sm.backward(numpynn_ce.backward()))

            # update the parameters
            torch_optimizer.step()
            numpynn_fc_net.update_parameters(lr)

    loss_diffs = torch_losses-numpynn_losses
    assert (loss_diffs < prec_thresh).all()

train_fc_classifiers(1,0.001,32)