from  ..numpynn.layers.fully_connected.fc import fc as fc
import numpy as np

rng = np.random.default_rng(seed=42)
batch_size = 2
nx = 2
n1 = 3
fc1 = fc.FullyConnectedLayer(nx,n1,batch_size,rng)

input = rng.random((nx,batch_size))
error = rng.random((n1,batch_size))

print("forward out:\n", fc1.forward(input))
print("\nbackward out:\n", fc1.backward(error))
fc1.update_parameters(1)