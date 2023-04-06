import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        # self.model = [
        #     # ConvolutionalLayer(in_channels=input_shape[2], out_channels=conv1_channels,
        #     #     filter_size=3, padding=1),
        #     ReLULayer(),
        #     MaxPoolingLayer(pool_size=2, stride=2),
        #     # ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels,
        #     # filter_size=3, padding=1),
        #     Flattener(),
        #     FullyConnectedLayer(n_input=16*16*3, n_output=n_output_classes)
        # ]
        # self.model = [
    
        #     Flattener(),
        #     FullyConnectedLayer(n_input=32*32*3, n_output=100),
        #     ReLULayer(),
        #     FullyConnectedLayer(n_input=100, n_output=10),

        # ]

        self.model = [
          Flattener(),
          FullyConnectedLayer(32*32*3, 100),
          ReLULayer(),
          FullyConnectedLayer(100, 10)

        ]
    
        # gradients pass
        # self.model = [
        #     ReLULayer(),
        #     MaxPoolingLayer(pool_size=2, stride=2),
        #     Flattener(),
        #     FullyConnectedLayer(n_input=16*16*3, n_output=n_output_classes)
        # ]
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for layer in self.model:
          layer.reset_grad()

        X_ = X.copy()
        for layer in self.model:
          X_ = layer.forward(X_)

        loss, d_pred = softmax_with_cross_entropy(X_, y)

        grad_out = d_pred.copy()
        for layer in reversed(self.model):
          grad_out = layer.backward(grad_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        X_ = X.copy()
        for layer in self.model:
          X_ = layer.forward(X_)
        
        print(X_)
        print(X_.argmax(axis=1))
        print('\n')
        return X_.argmax(axis=1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters

        for num, layer in enumerate(self.model):
          for param_name, param in layer.params().items():
            result[f'{num, param_name}'] = param

        return result