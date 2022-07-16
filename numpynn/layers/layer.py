'''
Layers should follow a modular structure by inheriting from
the abstract base class 'Layer'.
'''


from abc import ABC, abstractmethod


class Layer(ABC):
    # forward propagation takes the output from the previous layer
    @abstractmethod
    def forward(self, previous_output):
        pass

    # back propagation takes the gradient from the upstream layer
    @abstractmethod
    def backward(self, upstream_gradient):
        pass

    # need to know learning rate for parameter update
    @abstractmethod
    def update_parameters(self, eta, reset=True):
        pass

    @abstractmethod
    def __str__(self):
        pass