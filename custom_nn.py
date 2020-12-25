# import libs
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib import image
from pandas import DataFrame
from PIL import Image
import time
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
            print(self.W)

            # empty weight gradient matrix
            self.dW = np.zeros((self.N, self.N_p))

            # initialize a bias column vector with N rows
            self.B = np.random.randn(self.N,1)/10
            print(self.B)

            # empty bias gradient vector
            self.dB = np.zeros((self.N,1))

            # column vector of weighted sums
            self.z_cache = np.zeros((self.N,1))

            # keep a running sum for the weights and biases
            self.dW_sum = np.zeros((self.N, self.N_p))
            self.dB_sum = np.zeros((self.N,1))

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


class Network:
    # pass in a list of numbers for the layers
    def __init__(self, layers):
        self.num_layers = len(layers)
        
        self.layers = []

        # add the input layer
        IL = InputLayer(layers[0])
        self.layers.append([None, IL])

        # add a fully connected and activation layer for each "layer"
        prev_layer_size = layers[0]
        for i in range(1,len(layers)):
            FC = FullyConnectedLayer(layers[i], prev_layer_size)
            SM = SigmoidLayer(layers[i])
            prev_layer_size = layers[i]
            self.layers.append([FC, SM])
    
    def forward_pass(self, input_data):
        # set the first activation to the input data
        activation = input_data
        #print("Input data: ", activation)

        # set the input layer activations to the input data
        self.layers[0][1].a_cache = activation

        # for each layer (not including the input) propagate through the FC then AF
        for layer in range(1, self.num_layers):
            activation = self.layers[layer][1].forward_propagation(self.layers[layer][0].forward_propagation(activation))
        # print(activation)
        return activation

    def backward_pass(self, expected_output):
        # get the first upstream gradient, the derivative of the cost function w/r/t the activation of the output
        upstream_gradient = self.layers[-1][1].a_cache - expected_output # dc_da
        #print("first upstream gradient", upstream_gradient)
        
        # we backward pass from the last layer to the first hidden layer, not the input layer so -1
        for layer in range(1, self.num_layers):
            # back pass through the sigmoid
            delta = self.layers[-layer][1].backward_propagation(upstream_gradient)
            #print("delta", delta)

            # get the previous layer activations
            previous_layer_activations = self.layers[-layer-1][1].a_cache

            # back pass through the fully connected layer
            upstream_gradient = self.layers[-layer][0].backward_propagation(previous_layer_activations, delta)
            #print("upstream gradient", upstream_gradient)
        
    # update teh parameters over the gradient of one mini batch
    def mini_batch_pass(self, eta, mini_batch):
        # forward and backward pass for all input samples, dw & db are calculated internally
        for input, output in mini_batch:
            self.forward_pass(input)
            self.backward_pass(output)

        # update the parameters
        for layer in range(1,self.num_layers):
            self.layers[layer][0].update_parameters(len(mini_batch), eta)
    
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
                print("Epoch {0}: {1} / {2}".format(j, self.test_network(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def test_network(self, test_data):
        # test_results looks like [index, output col vector]
        test_results = [(np.argmax(self.forward_pass(input)), output) for (input, output) in test_data]

        return sum(int(output[max_index] == 1) for (max_index, output) in test_results)

if __name__ == "__main__":
    # net = Network([2,1])
    # print(net.layers)
    # print("input layer",net.layers[0])

    # print("weights",net.layers[1][0].W)
    # print("biases",net.layers[1][0].B)

    # data = np.array([[1,1]]).transpose()
    # net.forward_pass(data)

    # output_labels = np.array([2])
    # net.backward_pass(output_labels)

    # print("dW",net.layers[1][0].dW)
    # print("dB",net.layers[1][0].dB)

    # print("dw",net.layers[0][0].dW)
    # print("db",net.layers[0][0].dB)
            
    #net = Network([2,1])
    
    # 2D dataset to classify
    # 100 points split into two clusters (centers) with 2 features (x,y coordinate) inside center_box
    # input is the list of points, output is the list of labels for each point
    sample_size = 500
    data_input, output = make_blobs(n_samples=sample_size, centers=3, n_features=2, center_box=(0,100), cluster_std=2.5)

    # more difficult data is moons or circles
    #data_input, output = make_moons(n_samples=sample_size, noise=0.1)
    #data_input, output = make_circles(n_samples=sample_size, noise=0.06, factor=0.1)

    # create a table from these inputs and outputs
    frame = DataFrame(dict(x_coord=data_input[:,0], y_coord=data_input[:,1], category=output))

    # these are the two category labels for the data points
    labels = {0:'blue', 1:'red', 2:'green'}
    fig, ax = plt.subplots()

    # group the data by the category (either a 1 or 0)
    groups = frame.groupby('category')

    # plot the data
    for key, group in groups:
        group.plot(ax=ax, kind='scatter', x='x_coord', y='y_coord', label=key, color=labels[key])
    plt.show(block=False)
    

    net = Network([2,3])
    output_labels = []
    for out in output:
        data = np.zeros((3,1))
        data[out] = 1
        output_labels.append(data)

    data = []
    data_input = data_input / 100 # normalize data
    for x,y in zip(data_input, output_labels):
        data.append((x.reshape(2,1),y))
    
    training_data = data[0:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]

    # x_in = float(input("x:"))
    # y_in = float(input("y:"))
    # coord = np.array([[x_in,y_in]]).transpose()
    # print(net.forward_pass(coord))
    net.stochastic_gradient_descent(training_data,100,100,0.1,test_data)
    #print(net.forward_pass(coord))
    x_in = float(input("x:"))
    y_in = float(input("y:"))
    coord = np.array([[x_in,y_in]]).transpose()
    print(net.forward_pass(coord))
    x_in = float(input("x:"))
    y_in = float(input("y:"))
    coord = np.array([[x_in,y_in]]).transpose()
    print(net.forward_pass(coord))
    


