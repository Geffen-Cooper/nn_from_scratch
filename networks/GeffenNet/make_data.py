import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib import image
from pandas import DataFrame
from PIL import Image
import time
import random
from mnist.loader import MNIST
import pickle

def gen_clusters():

    # 2D dataset to classify
    # 5000 points split into n clusters (centers) with 2 features (x,y coordinate) inside center_box
    # input is the list of points, output is the list of labels for each point
    sample_size = 5000
    data_input, output = make_blobs(n_samples=sample_size, centers=3, n_features=2, center_box=(0,500), cluster_std=15)

    # more difficult data is moons or circles
    #data_input, output = make_moons(n_samples=sample_size, noise=0.1)
    #data_input, output = make_circles(n_samples=sample_size, noise=0.06, factor=0.1)

    # create a table from these inputs and outputs
    frame = DataFrame(dict(x_coord=data_input[:,0], y_coord=data_input[:,1], category=output))

    # these are the category labels for the data points
    labels = {0:'blue', 1:'red', 2:'green'}
    fig, ax = plt.subplots()

    # group the data by the category (either a 1 or 0)
    groups = frame.groupby('category')

    # plot the data
    for key, group in groups:
        group.plot(ax=ax, kind='scatter', x='x_coord', y='y_coord', label=key, color=labels[key])
    plt.show(block=False)

    output_labels = []
    for out in output:
        data = np.zeros((3,1))
        data[out] = 1
        output_labels.append(data)

    data = []
    data_input = data_input / 500 # normalize data
    for x,y in zip(data_input, output_labels):
        data.append((x.reshape(2,1),y))
    
    training_data = data[0:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    choices = ['blue', 'red', 'green']

    return (training_data, test_data, choices)

def gen_moons():
    # 2D dataset to classify
    sample_size = 7500
    data_input, output = make_moons(n_samples=sample_size, noise=0.075)
    #data_input, output = make_circles(n_samples=sample_size, noise=0.06, factor=0.1)

    # create a table from these inputs and outputs
    frame = DataFrame(dict(x_coord=data_input[:,0], y_coord=data_input[:,1], category=output))

    # these are the category labels for the data points
    labels = {0:'blue', 1:'red'}
    fig, ax = plt.subplots()

    # group the data by the category (either a 1 or 0)
    groups = frame.groupby('category')

    # plot the data
    for key, group in groups:
        group.plot(ax=ax, kind='scatter', x='x_coord', y='y_coord', label=key, color=labels[key])
    plt.show(block=False)

    output_labels = []
    for out in output:
        data = np.zeros((2,1))
        data[out] = 1
        output_labels.append(data)

    data = []
    data_input = data_input / 2 # normalize data
    for x,y in zip(data_input, output_labels):
        data.append((x.reshape(2,1),y))
    
    training_data = data[0:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    choices = ['blue', 'red']

    return (training_data, test_data, choices)

def gen_circles():
    # 2D dataset to classify
    sample_size = 5000
    data_input, output = make_circles(n_samples=sample_size, noise=0.06, factor=0.1)

    # create a table from these inputs and outputs
    frame = DataFrame(dict(x_coord=data_input[:,0], y_coord=data_input[:,1], category=output))

    # these are the category labels for the data points
    labels = {0:'blue', 1:'red'}
    fig, ax = plt.subplots()

    # group the data by the category (either a 1 or 0)
    groups = frame.groupby('category')

    # plot the data
    for key, group in groups:
        group.plot(ax=ax, kind='scatter', x='x_coord', y='y_coord', label=key, color=labels[key])
    plt.show(block=False)

    output_labels = []
    for out in output:
        data = np.zeros((2,1))
        data[out] = 1
        output_labels.append(data)

    data = []
    data_input += 5
    data_input = data_input/5
    for x,y in zip(data_input, output_labels):
        data.append((x.reshape(2,1),y))
    
    training_data = data[0:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    choices = ['blue', 'red']

    print(training_data[0:10])

    return (training_data, test_data, choices)

def get_mnist():
    mnist = MNIST('../../data/MNIST')
    print("Loading Data ... ")
    x_train, y_train = mnist.load_training() #60000 samples
    x_test, y_test = mnist.load_testing()    #10000 samples

    x_train = np.asarray(x_train).astype(np.float32)/255
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)/255
    y_test = np.asarray(y_test).astype(np.int32)

    print("formatting data")
    training_labels = []
    for y in y_train:
        label = np.zeros((10,1))
        label[y] = 1
        training_labels.append(label)

    testing_labels = []
    for y in y_test:
        label = np.zeros((10,1))
        label[y] = 1
        testing_labels.append(label)
    
    training_data = []
    for x,y in zip(x_train, training_labels):
        training_data.append((x.reshape(784,1),y))
    
    testing_data = []
    for x,y in zip(x_test, testing_labels):
        testing_data.append((x.reshape(784,1),y))

    return (training_data, testing_data)

    