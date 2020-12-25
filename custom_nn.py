# import libs
import numpy as np

# the fully connected layer calculates the weighted sum (the "z" neurons)
# it will need to keep track of the weights, biases, and partial derivatives
class FullyConnectedLayer:

    # need to know the number of neurons in this layer and previous layer to form weight matrix
    def __init__(self, layer_size, previous_layer_size):
        if previous_layer_size != None:
            self.N = layer_size
            self.N_p = previous_layer_size

            # initialize a weight matrix with N rows and N_p columns
            self.W = np.random.randn(self.N, self.N_p)

            # empty weight gradient matrix
            self.dW = np.zeros((self.N, self.N_p))

            # initialize a bias column vector with N rows
            self.B = np.random.randn(self.N,1)

            # empty bias gradient vector
            self.dB = np.zeros((self.N,1))

            # column vector of weighted sums
            self.z_cache = np.zeros((self.N,1))

    # do a weight matrix activation vector multiplication to get the weighted sum for the layer
    def forward_propagation(self, previous_layer_activations):
        # get the weighted sum for this layer
        if self.W.shape[1] != previous_layer_activations.shape[0]:
            print("mismatch between weight matrix columns and activation rows")
            print(self.W.shape[1],  previous_layer_activations.shape[0])
        else:
            self.z_cache = np.dot(self.W, previous_layer_activations) + self.B
        return self.z_cache

    # this calculates the partial derivatives for a backward pass and are overwritten on each call
    # when doing SGD we will need to add these values to some running count and then divide by the mini batch size
    def backward_propagation(self, pervious_layer_activations, upstream_gradient):
        # first calculate the partial derivatives for the weights, represents an outer product of two vectors
        self.dW = np.dot(upstream_gradient, pervious_layer_activations.transpose())

        # next set the partial derivatives for the biases to the upstream gradient
        self.dB = upstream_gradient

        # multiply the local gradient by the upstream and return this as the new upstream gradient
        return np.dot(self.W.transpose(), upstream_gradient)

    # once the average gradients are computed over a mini-batch, adjust the weights and biases
    def update_parameters(self, dW, dB, eta):
        self.W -= eta*dW
        self.B -= eta*dB

class SigmoidLayer:
    def __init__(self, layer_size):
        self.N = layer_size

        # column vector of activations
        self.a_cache = np.zeros((self.N,1))
    
    # apply the activation function on the weighted sum
    def forward_propagation(self, weighted_sum):
        self.a_cache = 1 / (1 + np.exp(-weighted_sum))
        return self.a_cache

    # multiply elementwise the derivative of the activation function by the upstream gradient and return
    def backward_propagation(self, upstream_gradient):
        sigmoid_prime = self.a_cache*(1-self.a_cache) 
        return np.multiply(sigmoid_prime, upstream_gradient)

# calculate cost for one sample
def quadratic_cost(expected_output, activation):
    return 1/2 * np.sum(np.square(expected_output-activation))

# derivatve of cost function
def quadratic_cost_prime(expected_output, activation):
    return (activation - expected_output)


class Network:
    # pass in a list of numbers for the layers
    def __init__(self, layers):
        self.num_layers = len(layers)
        
        self.layers = []

        # add a fully connected and activation layer for each "layer"
        prev_layer_size = layers[0]
        for i in range(1,len(layers)):
            FC = FullyConnectedLayer(layers[i], prev_layer_size)
            SM = SigmoidLayer(layers[i])
            prev_layer_size = layers[i]
            self.layers.append([FC, SM])
    
    def execute(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer[1].forward_propagation(layer[0].forward_propagation(activation))
        print(activation)

if __name__ == "__main__":
    net = Network([10,100,500,2])
    data = np.array([[1,1,1,1,1,1,1,1,1,1]]).transpose()
    net.execute(data)
            


