# This very simple one layer network tries to learn how 2D points are distributed
# For example, we have "blue" points that are clustered in the one part
# of the plane and "red" points that are clustered in the other part of the plane.
# The network should be able to learn these clusters and categorize a new point
# not in the given training set

# The network has two input nodes and an output node
# input: (x,y) coordinate 
# output: categorization, 1 or 0

# One issue with this network is that weigt and bias initialization is random.
# If the parameters initialize to relatively large values the sigmoid will saturate
# and the gradient won't change. Normalizing the data helped with this a lot.

# This network has a lot of difficulty on the circular data, may need more nodes/layers

# import libs
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib import image
from pandas import DataFrame
from PIL import Image
import time

# ================== Functions ===================

# activation functions and derivatives
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# activation functions and derivatives
def relu(x):
    return np.maximum(x,0)

def relu_derivative(x):
    return x > 0

# use MSE loss with 1/2 in front
def loss_derivative(a, y):
    return(a - y)

def predict(x, y):
    coord = np.array([x,y])
    result = sigmoid(np.dot(W, coord) + B)
    if(result < 0.5):
        print("blue", (result))
    else:
        print("red", (result))


# =========== Generate and visualize data ===============

# 2D dataset to classify
# 100 points split into two clusters (centers) with 2 features (x,y coordinate) inside center_box
# input is the list of points, output is the list of labels for each point
sample_size = 100
#data_input, output = make_blobs(n_samples=sample_size, centers=2, n_features=2, center_box=(0,100), cluster_std=2.5)

# more difficult data is moons or circles
data_input, output = make_moons(n_samples=sample_size, noise=0.1)
#data_input, output = make_circles(n_samples=sample_size, noise=0.06, factor=0.1)

# create a table from these inputs and outputs
frame = DataFrame(dict(x_coord=data_input[:,0], y_coord=data_input[:,1], category=output))

# these are the two category labels for the data points
labels = {0:'blue', 1:'red'}
fig, ax = plt.subplots()

# group the data by the category (either a 1 or 0)
groups = frame.groupby('category')

# plot the data
for key, group in groups:
    group.plot(ax=ax, kind='scatter', x='x_coord', y='y_coord', label=key, color=labels[key])
plt.show(block=False)


# ============== neural network initialization ===================

# transpose the input into column vectors, normalize data to avoid saturation
data_input = data_input.transpose()
data_input = data_input /2 # /100 for cluster, /2 for moons, /1 for circles

# transpose the output to a column vector
output = output.reshape(sample_size,1)


# randomize the weight matrix and bias vector
np.random.seed(int(time.time())%1000)
W = np.random.randn(1,2)
B = np.random.randn(1)
learning_rate = 0.1


# ============== Train the Network ================
for epoch in range(1000):
    # set the data_input
    X = data_input

    # ============= Forward Propagation ===============
    # Step 1: Dot product of inputs with weight matrix
    # note we do this for all the inputs at once
    z = np.dot(W, X) + B

    # Step 2: pass through the activation function
    # a is now a list of all the outputs
    a = sigmoid(z)

    # ============= Backward Propagation ===============
    # Back prop through the output layer
    loss = 0.5*(a.transpose()-output)**2
    print("loss: ", loss.sum()/sample_size)

    # derivative of cost with respect to output activations
    dc_da = loss_derivative(a.transpose(),output)

    # derivative of activation with respect to weighted sum
    da_dz = sigmoid_derivative(z)

    # call the intermediate value delta
    delta = np.multiply(dc_da.transpose(),da_dz)

    # derivative of weighted sum with respect to weights is the inputs

    # the gradient of the bias is the average over all the data
    db = np.average(delta)

    # the gradient of the weights is the same but multiplied by dz_dw
    dw = np.dot(data_input,delta.transpose())/sample_size

    B -= learning_rate*db
    W -= (learning_rate*dw).transpose()

# print the weights and biases as a sanity check
print("W: ", W)
print("B: ", B)

# test the network
x_in = float(input("x:"))
y_in = float(input("y:"))
predict(x_in, y_in)

