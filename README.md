# Simple Neural Network from Scratch  

This project is a set of python classes for building simple neural networks from scratch. Some of the code, ideas, and inspiration are taken from other sources which is explained and cited in the comments of *src/NeuralNet.py*.

# Sections  

[Project Structure](#project-structure)   
[Usage](#usage)  
[Examples](#examples)  
[Creating and Structuring Datasets](creating-and-structuring-datasets)  
[Code Explained](#code-explained)  

# Project Structure  

The project is split up into the following directories:  
**datasets**  
* contains MNIST dataset and some test images of my own handwriting  
* see [Usage and Examples](#usage-and-examples) for how to add more data

**plots**  
* plots of cost vs. epochs stored here  

**pre-trained**  
* pre-trained networks stored as .pickle files  
* see [Usage and Examples](#usage-and-examples) for how to save/load networks
  
**src**  
* source code where networks can be built, trained, tested  

# Usage  
### Setup
1. Clone the repo:  ```git clone https://github.com/geffencooper/nn_from_scratch.git```  
2. Install libraries: ```pip3 install [library]```
   * numpy
   * matplotlib (for plots)
   * python-mnist (for extracting MNIST data)
   * scikit-learn (for creating datasets)
   * Pillow (for plots)
   * pickle-mixin (for saving networks)
   * pandas (for datasets)
3. go to the src directory:  ```cd src```

### Building, Training, Testing
1. Create a *Network* object:  
```
net = Network(input_layer_size, cost_function, cost_function_derivative)
```
2. Add layers to the network:  
```
net.add_fc_layer(layer_size)
net.add_sigmoid_layer(layer_size)
net.add_fc_layer(layer_size)
net.add_sigmoid_layer(layer_size)
...
```
3. Train the network:
```
net.stochastic_gradient_descent(training_data, number_of_epochs, batch_size, learning_rate, testing_data)
```
```
console output:
---
```
4. Store the network to a file:
```
file_handle = open("file_name.pickle", "wb") # open a file
pickle.dump(net, file_handle) # store Network object to a file
file_handle.close() # close the file
```
5. Test the network on an image
```
img = Image.open("../datasets/images/img.png").convert('L') # load an image, convert to grayscale
img = np.asarray(img).astype(np.float32)/255.0 # convert image to numpy array and normalize
img = img.reshape(784,1) # convert array to column vector

file_handle = open("../pre-trained/network1.pickle", "rb") # open the file of the Network object
net = pickle.load(file_handle) # load the object
file_handle.close() # close the file

print("Result ....", np.argmax(net.forward_pass(img))) # pass the image through the network
```
# Examples
### Example 1: training and testing on MNIST dataset
1. Train on MNIST, *can also call train_mnist() inside of build_network.py*
```
training_data, testing_data = get_mnist() # get the training and testing data

# create a network
net = Network(784, quadratic_cost, quadratic_cost_prime)
net.add_fc_layer(50)
net.add_sigmoid_layer(50)
net.add_fc_layer(10)
net.add_sigmoid_layer(10)

# train the network
net.stochastic_gradient_descent(training_data,30,20,0.2,testing_data)

# store the network to a file
file_store = open("net_50_hidden_30_epochs.pickle", "wb")
pickle.dump(net, file_store)
file_store.close()
```
2. Test on an image of a digit (created by hand in microsoft paint, 28x28 px)
```
digit = Image.open("../datasets/my_handwriting/9.png").convert('L')
digit = np.asarray(digit).astype(np.float32)/255.0
digit = digit.reshape(784,1)

file_store = open("../pre-trained/net_50_hidden_30_epochs.pickle", "rb")
net = pickle.load(file_store)
file_store.close()

print("It is a ....", np.argmax(net.forward_pass(digit)))
```
### Example 2: Creating artificial data with scikit-learn
# Creating and Structuring Datasets

# Code Explained
