import numpy as np
import torch
import torch.nn as nn
import numpynn.layers.fully_connected.fc as fc

# global variables shared by both versions
rng = np.random.default_rng(seed=42)
batch_size = 2
nx = 2
n1 = 3

# let pytorch determine the parameters and input randomly
torch_fc = nn.Linear(nx,n1)
numpynn_fc = fc.FullyConnectedLayer(nx,n1,1,rng)
torch_input = torch.rand((nx,1))

# copy the values to numpynn
numpynn_fc.W = torch_fc.weight.detach().numpy()
numpynn_fc.B = torch_fc.bias.detach().numpy().reshape(numpynn_fc.B.shape)
numpynn_input = torch_input.detach().numpy()

def print_test():
    print("=== pytorch fc:===\n",torch_fc.weight,torch_fc.bias,torch_input)
    print("\n===numpynn fc:===\n",numpynn_fc,numpynn_input)

# test forward prop
def test_forward():
    assert (numpynn_fc.forward(numpynn_input) == torch_fc(torch_input.T).detach().numpy().T).all()

torch_pred = torch_fc(torch_input.T)
torch_label = torch.rand((3,1))
torch_loss = torch.sum(torch_label-torch_pred.T)
torch_loss.backward()

numpynn_pred = numpynn_fc.forward(numpynn_input)
numpynn_label = torch_label.detach().numpy()
numpynn_loss = numpynn_label - numpynn_pred
dL_dy = np.ones((n1,1))*-1
numpynn_fc.backward(dL_dy)

def test_backward():
    assert (numpynn_fc.dW == torch_fc.weight.grad.numpy()).all() and (numpynn_fc.dB == torch_fc.bias.grad.numpy()).all()