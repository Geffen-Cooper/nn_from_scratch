''' This file defines some helper functions shared by the tests'''

import torch
import torch.nn as nn
import numpy as np
from  numpynn.layers.fully_connected import fc as fc

# ===================================================================================== #
# ================================== Global Variables ================================= #
# ===================================================================================== #

# fix the seed when want to make test values deterministic
rng = np.random.default_rng(seed=42)

# define a precision threshold for comparing to pytorch
prec_thresh = 10**(-5)

''' ========================== Helper Functions used by tests ========================= '''


# ===================================================================================== #
# ==================================== Random Batch =================================== #
# ===================================================================================== #

# can be used to create random inputs and labels
# the convention for batches is each column is a sample
# dim[0] is number of features, dim[1] is batch size
def create_random_batch(dim):
    # let pytorch determine input and copy to numpynn
    torch_batch = torch.rand(dim)
    numpynn_batch = torch_batch.detach().numpy()

    return torch_batch, numpynn_batch


# ===================================================================================== #
# ==================================== Random fc layer ================================ #
# ===================================================================================== #

# creates a fully connected layer in torch and numpynn with the same initial parameters
def create_fc_layer(input_neurons, output_neurons):
    # let pytorch determine the parameters randomly
    torch_fc = nn.Linear(input_neurons,output_neurons)
    numpynn_fc = fc.FullyConnectedLayer(input_neurons,output_neurons,1,rng)

    # copy the values to numpynn
    numpynn_fc.W = torch_fc.weight.detach().numpy()
    numpynn_fc.B = torch_fc.bias.detach().numpy().reshape(numpynn_fc.B.shape)

    return torch_fc,numpynn_fc


# ===================================================================================== #
# ==================================== Random fc variables ============================ #
# ===================================================================================== #

# creates random layer variables
def create_fc_layer_variables():
    input_neurons = rng.integers(1,100)
    output_neurons = rng.integers(1,10)
    batch_size = 2**rng.integers(0,5)
    return input_neurons, output_neurons, batch_size


# ===================================================================================== #
# ==================================== Random Dimensions ============================== #
# ===================================================================================== #

# creates random layer dimension
def create_random_dimension(dimensions=2):
    # assume 2D inputs for now, TODO: for conv will be higher dimensionality
    input_neurons = rng.integers(1,100)
    batch_size = 2**rng.integers(0,5)
    return (input_neurons, batch_size)