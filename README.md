# Simple Neural Network from Scratch  

This project is a set of python classes for building simple neural networks from scratch. Some of the code, ideas, and inspiration are taken from other sources which is explained and cited in the comments of *src/NeuralNet.py*.

# Sections  

[Project Structure](#project-structure)   
[Usage](#usage)  
[Examples](#examples)  
[Structuring Datasets](#structuring-datasets)  
[Visualization](#visualization)  

# Project Structure  

The project is split up into the following directories:  
**datasets**  
* contains MNIST dataset and some test images of my own handwriting  
* see [Structuring Datasets](#structuring-datasets) for how to add more data

**plots**  
* plots of cost vs. epochs stored here  

**pre-trained**  
* pre-trained networks stored as .pickle files  
* see [Usage](#usage) for how to save/load networks
  
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
```ruby
net = Network(input_layer_size, cost_function, cost_function_derivative)
```
2. Add layers to the network:  
```ruby
net.add_fc_layer(layer_size)
net.add_sigmoid_layer(layer_size)
net.add_fc_layer(layer_size)
net.add_sigmoid_layer(layer_size)
...
```
3. Train the network:
```ruby
net.stochastic_gradient_descent(training_data, number_of_epochs, batch_size, learning_rate, testing_data)
```
```
console output:
---
```
4. Store the network to a file:
```ruby
file_handle = open("file_name.pickle", "wb") # open a file
pickle.dump(net, file_handle) # store Network object to a file
file_handle.close() # close the file
```
5. Test the network on an image
```ruby
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
```ruby
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
```ruby
digit = Image.open("../datasets/my_handwriting/9.png").convert('L')
digit = np.asarray(digit).astype(np.float32)/255.0
digit = digit.reshape(784,1)

file_store = open("../pre-trained/net_50_hidden_30_epochs.pickle", "rb")
net = pickle.load(file_store)
file_store.close()

print("It is a ....", np.argmax(net.forward_pass(digit)))
```
### Example 2: train on artificial cluster data from scikit-learn  
This example can be found in *build_network.py*
```ruby
# build the network layer by layer
net = Network(2, quadratic_cost, quadratic_cost_prime)
net.add_fc_layer(4)
net.add_sigmoid_layer(4)
net.add_fc_layer(3)
net.add_sigmoid_layer(3)

# generate the data
training_data, test_data, choices = gen_clusters()

# train the network
net.stochastic_gradient_descent(training_data,30,50,0.9,test_data)

# test the network by typing in a coordinate (can view from plot displayed)
# normalize the input data by dividing by 500

# input 1
x_in = float(input("x:"))/500
y_in = float(input("y:"))/500
coord = np.array([[x_in,y_in]]).transpose()
print(choices[np.argmax(net.forward_pass(coord))])

# input 2
x_in = float(input("x:"))/500
y_in = float(input("y:"))/500
coord = np.array([[x_in,y_in]]).transpose()
print(choices[np.argmax(net.forward_pass(coord))])

# input 3
x_in = float(input("x:"))/500
y_in = float(input("y:"))/500
coord = np.array([[x_in,y_in]]).transpose()
print(choices[np.argmax(net.forward_pass(coord))])
```

# Structuring Datasets
Training and test data are passed to the network as a list of tuples:

[ (input_data_1, output_label_1), (input_data_2, output_label_2), ... ]  

The shape of the input data is a column vector (nx1 numpy array)
The output data is a one-hot column vector
### Example
```ruby
dataset = []
input = np.zeros((5,1)) # create a 5x1 column vector
label = np.array([[0,0,1,0]]).transpose() # create a one-hot 4x1 column vector
dataset.append((input, label))
```
**See** *make_data.py* **for in more complex examples of how to create a dataset**


# Visualization
The classes in *NeuralNet.py* are created in a way that makes it easy to build and  
train small feed-forward networks for simple classification tasks and experimenting
with different hyperparameters. There are no convolution layers.   

### High Level Structure 
In *NeuralNet.py* there layer classes which we can generalize as follows:
* **input layer:** dummy layer to feed data sample into the network
* **fully connected layer:** computes weighted sum of previous layer outputs
* **activation layer:** passes previous layer output through activation
* **output:** compares final layer output to the known label  
![layers](readme_images/parts.png)  

In general the fully connected layer and activation layer come as a pair where the weighted sum is passed to the activation layer on a forward pass and the upstream gradient multiplied element-wise by the local derivative of the activation gets passed back to the fully connected layer on a backwards pass.  
![pass](readme_images/passes.png)  

A detailed example of a forward and backward pass by hand for a sigmoid activation and MSE cost is shown:
### Forward Pass for fully connected layer and activation layer
* the notation below treats the fully connected layer and activation layer as a pair layer "l" 
* in the code, these two layers are independent objects but the assumption is that an activation will always come after a fully connected layer  
![forward](readme_images/forward_pass.png)  

### Backward Pass for fully connected layer and sigmoid layer with MSE cost function  
* this diagram has a lot going on but the equations at the bottom describe the backward pass concisely
* for an activation layer, it multiplies the **upstream gradient** *elementwise* by the **local derivative of the activation function** evaluated at the **input** which is the **weighted sum** passed into it.
* for a fully connected layer, it does a *matrix multiplication* of the **weight matrix transpose** and the **upstream gradient** which is denoted by an intermediate value called *delta* in the diagram
* the reason we do this for a fully connected layer is because the derivative of the weighted sum with respect to the activation is the weights. However, the gradient will merge in the next layer back so the branches must be summed. So the weight matrix transpose multiplied by the upstream gradient accounts for the branching. This is understood more clearly when observing a single activation, such as a_1 in layer 2.
![detailed](readme_images/detailed.png)   
* now to get the partial derivatives of the weights we multiply the upstream gradient by the derivative of the weighted sum with respect to the weights, which is the activations. For the biases, we do the same thing but since the bias is multiplied by one, the derivative is 1.
* so dw_dc = upstream_grad times activations and db_dc = upstream_grad
* for the weights, this becomes an outer product so we get a matrix which is shown in purple