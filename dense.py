import numpy as np
from activation import Activation
from model import Model


class Dense:
    # TODO add multiple losses/activations

    def __init__(self, input_len, nodes, activation="softmax"):
        self.input_len = input_len
        self.nodes = nodes
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.random.randn(nodes) / nodes
        self.activations = np.zeros(nodes)
        self.activation = Model.activations[activation]
        self.activation_prime = Model.activations_prime[activation]
        self.input_shape = None
        self.input = None

    def forward(self, input):
        """
        Forward pass over the dense layer
        :param numpy.ndarray input: flattened numpy array of shape (height, width, feature maps)
        :return numpy.ndarray activations: numpy array of shape (nodes)
        """
        self.input_shape = input.shape
        self.input = input.flatten()
        weighted_sum = np.dot(self.input, self.weights) + self.biases
        self.activations = self.activation(weighted_sum)
        return self.activations

    def backward(self, d_loss_d_out, step):
        """
        Performs the backward pass of the convolution layer.
        :param numpy.ndarray d_loss_d_out: derivative of loss w.r.t the output of the forward pass. Same shape as the
        output of the forward pass
        :param float step: learning rate used in weight change. only here to keep params same for backward pass
        :return numpy.ndarray d_loss_d_input: derivative of loss w.r.t the input of the forward pass. Same shape as the
        input
        """
        # derivative of loss w.r.t weight = dl/do . do/ds . ds/dw
        d_out_d_summed_weights = self.activation_prime(self.activations)
        if len(d_out_d_summed_weights.shape) == 1:
            d_loss_d_summed_weights = d_out_d_summed_weights * d_loss_d_out
        else:
            d_loss_d_summed_weights = np.dot(d_out_d_summed_weights, d_loss_d_out)
        # ds/dw = input ( t = input * w + b)
        d_loss_d_weights = np.zeros((self.input_len, self.nodes))
        for i, input in enumerate(self.input):
            d_loss_d_weights[i] = input * d_loss_d_summed_weights

        # derivative of loss w.r.t bias = dl/ds . ds/db
        # ds/db = 1 ( s = input * w + b)
        # dl/db = dl/ds
        d_loss_d_bias = d_loss_d_summed_weights

        # update weights
        self.weights -= step * d_loss_d_weights
        self.biases -= step * d_loss_d_bias

        # derivative of loss w.r.t input = dl/ds . ds/di
        # ds/di = w ( t = input * w + b)
        d_loss_d_input = self.weights @ d_loss_d_summed_weights
        return d_loss_d_input.reshape(self.input_shape)


if __name__ == '__main__':
    dense = Dense(4, 10)
    dense.forward(np.array([1, 2, 3, 4]))
    print(dense.backward(np.array([0.0431151, -0.98447558, 0.02294952, 0.14973398, 0.00290415, 0.14513771, 0.13190235,
                                   0.00819373,  0.45531891,  0.02522013]), 0.05))
