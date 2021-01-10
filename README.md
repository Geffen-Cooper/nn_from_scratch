# Simple Neural Network from Scratch  

This project is a set of python classes for building simple neural networks from scratch. Some of the code, ideas, and inspiration are taken from other sources which is explained and cited in the comments of *src/NeuralNet.py*.

# Sections  

[Project Structure](#project-structure)  
[Usage and Examples](#usage-and-examples)  
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

# Usage and Examples  
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

### Building and Training
1. Create a *Network* object:  ``` net = Network(input_layer_size, cost_function, cost_function_derivative)```

# Code Explained
