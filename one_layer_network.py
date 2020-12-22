# This simple one layer network tries to learn how 2D points are distributed
# For example, we may have "blue" points that are clustered in the left part
# of the plane and "red" points that are clustered in the right part of the plane.
# The network should be able to learn these clusters and categorize a new point
# not in the given training set

# The network has two input nodes and an output node
# input: (x,y) coordinate 
# output: categorization, 1 or 0

# import libs
import numpy as np
from sklearn.datasets import make_blobs
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

def loss_derivative(a, y):
    return(a - y)


# =========== Generate and visualize data ===============

# 2D dataset to classify
# 100 points split into two clusters (centers) with 2 features (x,y coordinate) inside center_box
# input is the list of points, output is the list of labels for each point
sample_size = 100
data_input, output = make_blobs(n_samples=sample_size, centers=2, n_features=2, center_box=(0,100), cluster_std=2.5)

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

# transpose the input into column vectors
data_input = data_input.transpose()

# transpose the output to a column vector
output = output.reshape(sample_size,1)

# randomize the weight matrix and bias vector
np.random.seed(int(time.time())%1000)
W = np.random.randn(1,2)/1000
B = np.random.randn(1)/1000
learning_rate = 0.05

# ============== Train the Network ================
for epoch in range(5000):
    # set the data_input
    X = data_input

    # ============= Forward Propagation ===============
    # Step 1: Dot product of inputs with weight matrix
    # note we do this for all the inputs at once
    z = np.dot(W, X) + B
    #print("z: ", z)

    # Step 2: pass through the activation function
    # a is now a list of all the outputs
    a = sigmoid(z)

    # ============= Backward Propagation ===============
    # Step 1: Back prop through the output layer
    loss = 0.5*(a.transpose()-output)**2
    print("loss: ", loss.sum())

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

print("W: ", W)
print("B: ", B)

x_in = int(input("x:"))
y_in = int(input("y:"))
coord = np.array([x_in, y_in])

result = sigmoid(np.dot(W, coord) + B)

if(result < 0.5):
    print("blue", (result))
else:
    print("red", (result))

x_in = int(input("x:"))
y_in = int(input("y:"))
coord = np.array([x_in, y_in])

result = sigmoid(np.dot(W, coord) + B)

if(result < 0.5):
    print("blue", (result))
else:
    print("red", (result))

x_in = int(input("x:"))
y_in = int(input("y:"))
coord = np.array([x_in, y_in])

result = sigmoid(np.dot(W, coord) + B)

if(result < 0.5):
    print("blue", (result))
else:
    print("red", (result))