import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        raise Exception("Not implemented!")        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        # Initialize the output tensor
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # Apply padding to the input tensor
        X_padded = np.pad(X, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), mode='constant')
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                receptive_field = X_padded[:, y:y+self.filter_size, x:x+self.filter_size, :, np.newaxis]
                # Reshape tensor to matrix
                field_vc_len = int(np.product(receptive_field.shape) / batch_size)
                w_col_len = int(np.product(self.W.value.shape) / field_vc_len)
                receptive_field_matrix = receptive_field.reshape(batch_size, field_vc_len)
                W_matrix = self.W.value.reshape(field_vc_len, w_col_len)

                val = np.dot(receptive_field_matrix, W_matrix) + self.B.value

                output[:, y, x, :] = val
                # output1[:, y, x, :] = np.sum(receptive_field * self.W.value[np.newaxis, :, :, :], axis=(1,2,3)) + self.B.value
                # TODO: Implement forward pass for specific location
        
        self.X = X
        self.X_padded = X_padded

        return output

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        # Initialize the output tensors for gradients
        d_input = np.zeros_like(self.X)
        d_W = np.zeros_like(self.W.value)
        d_B = np.zeros_like(self.B.value)

        # Apply padding to the input tensor
        d_X_padded = np.pad(d_input, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)), mode='constant')

        for y in range(out_height):
            for x in range(out_width):
                receptive_field = self.X_padded[:, y:y+self.filter_size, x:x+self.filter_size, :, np.newaxis]
                # Reshape tensor to matrix
                field_vc_len = int(np.product(receptive_field.shape) / batch_size)
                w_col_len = int(np.product(self.W.value.shape) / field_vc_len)
                receptive_field_matrix = receptive_field.reshape(batch_size, field_vc_len)
                W_matrix = self.W.value.reshape(field_vc_len, w_col_len)

                # # Calculate gradients of loss w.r.t. parameters (weights and biases)
                d_val = d_out[:, y, x, :]
                d_W_matrix = np.dot(receptive_field_matrix.T, d_val)
                d_B_val = np.sum(d_val, axis=0)

                # Calculate gradients of loss w.r.t. input
                d_receptive_field_matrix = np.dot(d_val, W_matrix.T)
                d_receptive_field = d_receptive_field_matrix.reshape(receptive_field.shape)

                d_X_padded[:, y:y+self.filter_size, x:x+self.filter_size, :] += d_receptive_field[..., 0]

                # Update gradients of parameters
                d_W += d_W_matrix.reshape(self.W.value.shape)
                d_B += d_B_val

        # Remove padding from the gradients of input
        if self.padding > 0:
            d_input = d_X_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_input = d_X_padded
        # Update the parameters of the layer
        self.W.grad += d_W
        self.B.grad += d_B

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
    
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape

        output_height = int((height - self.pool_size) / self.stride + 1)
        output_width = int((width - self.pool_size) / self.stride + 1)
        
        output = np.zeros((batch_size, output_height, output_width, channels))
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        for y in range(output_height):
            for x in range(output_width):
                window = X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size, :]

                output[:, x, y, :] = np.max(window, axis=(1, 2))

        return output


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape

        output_height = int((height - self.pool_size) / self.stride + 1)
        output_width = int((width - self.pool_size) / self.stride + 1)
        
        dX = np.zeros_like(self.X)

        for y in range(output_height):
            for x in range(output_width):
                window = self.X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size, :]
                max_vals = np.amax(window, axis=(1, 2), keepdims=True)
                max_mask = (window == max_vals)
                dX[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size, :] = max_mask * (d_out[:, y, x, :])[:, None, None, :]
        return dX
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
