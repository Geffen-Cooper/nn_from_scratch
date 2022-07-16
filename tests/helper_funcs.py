''' This file defines some helper functions shared by the tests'''

import torch
import torch.nn as nn
import numpy as np

# ===================================================================================== #
# ================================== Global Variables ================================= #
# ===================================================================================== #

rng = np.random.default_rng(seed=42)
prec_thresh = 10**(-5)

''' ========================== Helper Functions used by tests ========================= '''


# ===================================================================================== #
# ==================================== Random Batch =================================== #
# ===================================================================================== #

# can be used to create random inputs and labels
def create_random_batch(dimension):
    # let pytorch determine input and copy to numpynn
    torch_batch = torch.rand(dimension)
    numpynn_batch = torch_batch.detach().numpy()

    return torch_batch, numpynn_batch