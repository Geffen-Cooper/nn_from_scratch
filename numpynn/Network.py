'''
Networks should follow a modular structure by inheriting from
the abstract base class 'Network'.
'''

class Network():
    def __init__(self):
        self.layers = []

    # layers can be added like this because of the modular structure enforced by Layer.py
    def add_layer(self,layer):
        self.layers.append(layer)

    # does a forward pass through the model
    def forward_pass(self, X):
        output = X

        # for each layer do forward propagation
        for layer in self.layers:
            output = layer.forward(output)

        return output

    # does a backward pass through the model, pass in the gradient of the loss w/r/t the last activation (dL_dY)
    def backward_pass(self, upstream_gradient):
        # we backward pass from the last layer to the first hidden layer
        for layer in reversed(self.layers):
            upstream_gradient = layer.backward(upstream_gradient)

    # need to know learning rate for parameter update
    def update_parameters(self, eta):
        # update the parameters for each layer after a backwards pass
        for layer in self.layers:
            layer.update_parameters(eta,reset=True)

    def __str__(self):
        pass