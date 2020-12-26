# import libs
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib import image
from pandas import DataFrame
from PIL import Image
import time
import random

class Network:

    # constructor
    """ pass in a list called layers that says the number of neurons in the 
        respective layer of the network. For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer containing
        2 neurons, the second layer 3 neurons, and the third layer 1 neuron. The
        biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first layer is assumed
        to be an input layer, and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs from later layers."""
    def __init__(self, sizes):

        self.num_layers = len(sizes)

        # this list is the neurons per layer
        self.sizes = sizes

        # biases is represented as a list of column vectors which are the biases for each layer, excluding the input
        # np.random.randn() takes in (... d2, d1, d0) as dimensions
        #   d0 is the number of elements (cols)
        #   d1 is the number of arrays (rows)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # weights is represented as a list of matrices with each matrix holding the weights for each layer
        # zip means to iterate over two lists in parallel and to stop when reach last index of shortest list
        # this for loop is iterating over connections between neurons in layer l and l + 1
        # say sizes was [2,3,1]
        # the network looks like:     n 
        #                        n 
        #                             n   n
        #                        n  
        #                             n
        # sizes[:-1] --> [2,3]        
        # sizes[1:]  --> [3,1]
        # so y,x gets the pairs (3,2) and (1,3) and makes randn numpy matrices
        # so a matrix with 2 cols and 3 rows and 3 cols and 3 rows
        # that means 3 sets of 2 connections and 1 set of 3 connections representing
        # how neurons in layer l connect to neurons in layer l-1
        # each element in weights is a weight matrix for cconnections between layer l and l-1
        # the elements in a row represent how how a neuron in layer l is connected to all neurons in l-1
        # the number of rows is the number of neurons in layer l
        # the number of cols is the number of neurons in layer l-1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # run through the network, used to get the output
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        
        # here we go layer by layer doing a matrix vector multiplication of the weights and activations
        # then the biases are added and the sigmoid function is applied
        # nothing about the network is changed here, and the activations of each layer are not
        # saved but rather calculated on the fly over and over again based on the layer before
        # the final activation is the output layer which gets returned
        # Remember that the number of elements in biases and weights are the number of layers
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # the driver function that runs through the epochs
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        # if test data is provided, then the network will be evaluated during training
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)
        
        # for every epoch do the following:
        # shuffle the training data, create the mini batches, do GD for each mini batch, test the network
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
                self.update_mini_batch(mini_batch, eta)
                
            # if have test data, then evaluate the network at each epoch    
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


    # this is where gradient descent is done on a mini batch
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
        # nabla is the gradient symbol
        # here we create empty matrix/vector copies of self.biases and self.weights to store the
        # partial derivatives
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # each item in the mini batch is a tuple of the input data and desired output
        for x, y in mini_batch:
            # get the partial derivatives of the weights and biases for each x in mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            # add all these partial derivatives together for a mini batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # now find the average of the partial derivatives, "avg gradient" for the mini batch
        # multiply by the learning rate and then subtract from the weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # this function calculates the gradient for a specific training example
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        # create the partial derivative storage matrix/vector
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        # feed forward through whole network, storing intermediate z and activation values
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        # this is the "delta" errors for the output layer, it is a column vector
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        
        # the gradient of the biases for the output layer is just the delta errors
        nabla_b[-1] = delta
        
        # the gradient of the weights for the output layer is the delta error 
        # times the activation of the layer before the last layer   
        # the reason this is a dot product is because we are dotting two
        # vectors to get a matrix of weights
        # w0,0 = e0*a0
        # w1,0 = e1*a0
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        # iterate over layers in reverse starting at second to last layer
        for l in range(2, self.num_layers):
            z = zs[-l] # get the z's of the lth to last layer
            sp = sigmoid_prime(z) # calculate the sigmoid prime of it
            
            # the delta error vector is the weights transposed multiplied by the last delta error
            # times the sigmoid prime
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            
            # add delta to biases gradient
            nabla_b[-l] = delta
            
            # see comment above for output layer
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        # feedforward runs an input through the network and returns the output vector
        # argmax compares the output vector elements with the expected output y and returns
        # the index of the output with the highest probability is returned
        # this is done for all the test data and put intot the list test_results
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
        # test result is (network's guess, [0, 0, 0, 1])
        
        # if the output equals the expected output then the sum is incremented
        return sum(int(y[x] == 1) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == "__main__":
    net = Network([2,3,1])
    
    test_x = np.array([1,2])
    test_x.shape = (2,1)
    
    test_y = np.array([2,3])
    test_y.shape = (2,1)