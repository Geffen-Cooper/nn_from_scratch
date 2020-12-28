''' This is a network is built using a modular layer structure using the ideas
    of computational graphs for backpropagation. Most of the code is original
    but some, including the high level control flow and the test code, is taken
    from Michael Nielson's Neural Networks and Deep Learning Book''' 

# import libs
import numpy as np
import random

# the fully connected layer calculates the weighted sum (the "z" neurons)
# it will need to keep track of the weights, biases, and partial derivatives
class FullyConnectedLayer:

    # need to know the number of neurons in this layer and previous layer to form weight matrix
    def __init__(self, layer_size, previous_layer_size):
        if previous_layer_size != None:
            self.N = layer_size
            self.N_p = previous_layer_size

            # initialize a weight matrix with N rows and N_p columns
            self.W = np.random.randn(self.N, self.N_p)/10
            # print(self.W)

            # empty weight gradient matrix
            self.dW = np.zeros((self.N, self.N_p))

            # initialize a bias column vector with N rows
            self.B = np.random.randn(self.N,1)/10
            # print(self.B)

            # empty bias gradient vector
            self.dB = np.zeros((self.N,1))

            # column vector of weighted sums
            self.z_cache = np.zeros((self.N,1))

            # keep a running sum for the weights and biases
            self.dW_sum = np.zeros((self.N, self.N_p))
            self.dB_sum = np.zeros((self.N,1))

            # store the previous layer activations
            self.previous_layer_activations = np.zeros((previous_layer_size,1))

    # do a weight matrix-activation vector multiplication to get the weighted sum for the layer
    def forward_propagation(self, previous_layer_activations):
        # get the weighted sum for this layer
        if self.W.shape[1] != previous_layer_activations.shape[0]:
            print("mismatch between weight matrix columns and activation rows")
            print(self.W.shape[1],  previous_layer_activations.shape[0])
            exit()
        else:
            self.z_cache = np.dot(self.W, previous_layer_activations) + self.B
            self.previous_layer_activations = previous_layer_activations
        return self.z_cache

    # this calculates the partial derivatives for a backward pass
    def backward_propagation(self, upstream_gradient):
        # first calculate the partial derivatives for the weights, represents an outer product of two vectors
        self.dW = np.dot(upstream_gradient, self.previous_layer_activations.transpose())
        self.dW_sum += self.dW

        # next set the partial derivatives for the biases to the upstream gradient
        self.dB = upstream_gradient
        self.dB_sum += self.dB

        # dot the local gradient by the upstream and return this as the new upstream gradient
        # this accounts for branching when two downstream gradients merge 
        return np.dot(self.W.transpose(), upstream_gradient)

    # once a mini-batch finishes, adjust the parameters by the average gradient over the mini-batch
    def update_parameters(self, mini_batch_size, eta):
        self.W -= eta*(self.dW_sum / mini_batch_size)
        self.B -= eta*(self.dB_sum / mini_batch_size)

        self.dW_sum[:] = 0
        self.dB_sum[:] = 0

# the sigmoid layer is a nonlinearity activation function layer
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

    def update_parameters(self, mini_batch_size, eta):
        pass

# the sigmoid layer is a nonlinearity activation function layer
class ReluLayer:
    def __init__(self, layer_size):
        self.N = layer_size

        # column vector of activations
        self.a_cache = np.zeros((self.N,1))
    
    # apply the activation function on the weighted sum
    def forward_propagation(self, weighted_sum):
        self.a_cache = np.maximum(weighted_sum,0)
        return self.a_cache

    # multiply elementwise the derivative of the activation function by the upstream gradient and return
    def backward_propagation(self, upstream_gradient):
        relu_prime = (self.a_cache > 0)
        return np.multiply(relu_prime, upstream_gradient)

    def update_parameters(self, mini_batch_size, eta):
        pass

# this is a dummy layer used for the input to keep modular structure
class InputLayer:
    def __init__(self, layer_size):
        self.N = layer_size

        # column vector of activations
        self.a_cache = np.zeros((self.N,1))

    def update_parameters(self, mini_batch_size, eta):
        pass

# calculate cost for one sample
def quadratic_cost(expected_output, activation):
    return 1/2 * np.sum(np.square(expected_output-activation))

# derivatve of cost function
def quadratic_cost_prime(expected_output, activation):
    return (activation - expected_output)

# calculate cost for one sample
def cross_entropy_cost(expected_output, activation):
    pass
    #return 1/2 * np.sum(np.square(expected_output-activation))

# derivatve of cost function
def cross_entropy_cost_prime(expected_output, activation):
    pass
    #return (activation - expected_output)


class Network:
    # pass in a list of numbers for the layers
    def __init__(self, input_size, cost_prime):
        self.cost_prime = cost_prime
        self.layers = []

        # add the input layer
        IL = InputLayer(input_size)
        self.layers.append(IL)

        self.prev_layer_size = input_size
    
    def add_fc_layer(self, layer_size):
        self.layers.append(FullyConnectedLayer(layer_size, self.prev_layer_size))
        self.prev_layer_size = layer_size

    def add_sigmoid_layer(self, layer_size):
        self.layers.append(SigmoidLayer(layer_size))
        self.prev_layer_size = layer_size

    def add_relu_layer(self, layer_size):
        self.layers.append(ReluLayer(layer_size))
        self.prev_layer_size = layer_size
    
    def forward_pass(self, input_data):
        # set the input layer activations to the input data
        self.layers[0].a_cache = input_data
        output = input_data

        # for each layer (not including the input) do forward propagation
        for layer in range(1, len(self.layers)):
            output = self.layers[layer].forward_propagation(output)
        # print(output)
        return output

    def backward_pass(self, expected_output):
        # get the first upstream gradient, the derivative of the cost function w/r/t the activation of the output
        #upstream_gradient = self.layers[-1].a_cache - expected_output # dc_da
        upstream_gradient = self.cost_prime(expected_output, self.layers[-1].a_cache)

        # print("first upstream gradient", upstream_gradient)
        
        # we backward pass from the last layer to the first hidden layer, not the input layer so -1
        for layer in range(1, len(self.layers)):
            # back pass through the sigmoid
            upstream_gradient = self.layers[-layer].backward_propagation(upstream_gradient)
        #print("upstream gradient", upstream_gradient)
        
    # update teh parameters over the gradient of one mini batch
    def mini_batch_pass(self, eta, mini_batch):
        # forward and backward pass for all input samples, dw & db are calculated internally
        for input, output in mini_batch:
            self.forward_pass(input)
            self.backward_pass(output)

        # update the parameters
        for layer in range(1, len(self.layers)):
            self.layers[layer].update_parameters(len(mini_batch), eta)
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # if test data is provided, then the network will be evaluated during training
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data) # shuffle the training data
            
            # create a list of mini batches by splitting the training data in intervals of mini_batch_size
            # n is the number of items in the training data
            # here range means to iterate from 0 to n by interval of mini_batch_size
            # so mini_bathes[0] = training_data[0:mini_batch_size]
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
                ]
           
            # once the mini batches are created update the network w's and b's by 
            # doing gradient descent for each mini batch
            for mini_batch in mini_batches:
                self.mini_batch_pass(eta, mini_batch)
                
            # if have test data, then evaluate the network at each epoch    
            if test_data:
                #print("Epoch {0}: Train_data: {1} / {2}".format(j, self.test_network(training_data), len(training_data)))
                print("Epoch {0}: Test_data: {1} / {2}".format(j, self.test_network(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def test_network(self, test_data):
        # test_results looks like [index, output col vector]
        test_results = [(np.argmax(self.forward_pass(input)), output) for (input, output) in test_data]

        return sum(int(output[max_index] == 1) for (max_index, output) in test_results)