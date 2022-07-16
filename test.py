import numpynn.layers.fully_connected.fc as fc
import numpy as np

rng = np.random.default_rng(seed=42)
batch_size = 2
nx = 2
n1 = 3
fc1 = fc.FullyConnectedLayer(nx,n1,batch_size,rng)

input = rng.random((nx,batch_size))

print(fc1.forward(input))