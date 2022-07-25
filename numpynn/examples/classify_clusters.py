''' This file runs tests on full training of a fc net '''

import sys
sys.path.append("../") # TODO: find a way to avoid this
sys.path.append("../..") # TODO: find a way to avoid this
from layers.relu import relu as relu
from layers.tanh import tanh as tanh
from  layers.cross_entropy import cross_entropy as ce
from layers.softmax import softmax as sm
from  layers.fully_connected import fc as fc
from numpynn import Network as net
from data.generate_data import *
import torch
import torch.nn as nn
import numpy as np
from tests.helper_funcs import *

# first define the pytorch model
class TorchFCNet(nn.Module):
    def __init__(self,input_neurons, output_neurons):
        super(TorchFCNet, self).__init__()
        # list of layers
        self.layers = nn.ModuleList()

        self.fc1 = nn.Linear(input_neurons,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,output_neurons)
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)
        self.layers.append(self.fc3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
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
    torch_fc_net, numpynn_fc_net = create_fc_nets(2,5)

    # create the data
    X_train,Y_train,X_test,Y_test = gen_clusters(test_split=0.2,num_clusters=5)
    num_batches = X_train.shape[0] // batch_size

    X_train = X_train / 500
    X_test = X_test / 500

    # create the loss functions
    torch_smce = nn.CrossEntropyLoss()
    numpynn_sm = sm.SoftmaxLayer()
    numpynn_ce = ce.CrossEntropyLayer()

    # torch optimizer
    torch_optimizer = torch.optim.SGD(torch_fc_net.parameters(),lr=lr)

    torch_losses = np.zeros(num_batches*epochs)
    numpynn_losses = np.zeros(num_batches*epochs)

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
            
            torch_losses[epoch*(num_batches) + batch_idx] = torch_loss.detach().item()
            numpynn_losses[epoch*(num_batches) + batch_idx] = numpynn_loss
            # backward pass
            torch_loss.backward()
            numpynn_fc_net.backward_pass(numpynn_sm.backward(numpynn_ce.backward()))

            # update the parameters
            torch_optimizer.step()
            numpynn_fc_net.update_parameters(lr)

    loss_diffs = torch_losses-numpynn_losses
    assert (loss_diffs < prec_thresh).all()

    plt.plot(numpynn_losses)
    print("avg loss:",np.sum(numpynn_losses[-num_batches:])/num_batches)
    plt.title("Training Loss")
    plt.xlabel("Batch Iteration")
    plt.ylabel("Batch Loss (Average)")
    plt.show()
    

    # go through test data
    torch_fc_net.eval()
    num_batches = X_test.shape[0] // batch_size
    torch_preds = np.zeros(X_test.shape[0])
    numpynn_preds = np.zeros(X_test.shape[0])
    for batch_idx in range(num_batches):
        # forward pass
        start = batch_idx*batch_size
        end = start + batch_size
        torch_out = torch_fc_net(torch.from_numpy(X_test[start:end]).float())
        numpynn_out = numpynn_fc_net.forward_pass(X_test[start:end].T)

        torch_preds[start:end] = np.argmax(torch_out.detach().numpy(),axis=1)
        numpynn_preds[start:end] = np.argmax(numpynn_out,axis=0)
   
    print(np.sum(torch_preds == Y_test)/len(Y_test))
    # x_wrong = X_test[torch_preds != Y_test]
    # y_wrong = Y_test[torch_preds != Y_test]
    wrong_idxs = np.where(torch_preds != Y_test)[0]
    show_clusters(X_train,Y_train,X_test,Y_test,torch_fc_net,wrong_idxs)

train_fc_classifiers(50,0.1,64)