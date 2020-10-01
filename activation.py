import numpy as np


class Activation:

    @staticmethod
    def sigmoid(weighted_sum):
        """
        Activation using sigmoid function
        :param numpy.ndarray weighted_sum: nump array of the weighted sums for each node in a layer
        :return numpy.ndarray activations: numpy array of the activation of the nodes in the layer
        """
        return 1 / (1 + np.exp(-weighted_sum))

    @staticmethod
    def sigmoid_prime(activations):
        """
        Derivative of the sigmoid activation function
        :param numpy.ndarray activations: nump array of the activations for each node in a layer
        :return numpy.ndarray derivatives: numpy array of the activation derivatives of the nodes in the layer
        """
        return activations * (1 - activations)

    @staticmethod
    def tanh(weighted_sum):
        """
        Activation using tanh function
        :param numpy.ndarray weighted_sum: nump array of the weighted sums for each node in a layer
        :return numpy.ndarray activations: numpy array of the activation of the nodes in the layer
        """
        return (np.exp(weighted_sum) - np.exp(-weighted_sum)) / (np.exp(weighted_sum) + np.exp(-weighted_sum))

    @staticmethod
    def tanh_prime(activations):
        """
        Derivative of the sigmoid activation function
        :param numpy.ndarray activations: nump array of the activations for each node in a layer
        :return numpy.ndarray derivatives: numpy array of the activation derivatives of the nodes in the layer
        """
        return 1 - (activations ** 2)

    @staticmethod
    def relu(weighted_sum):
        """
        Activation using relu function
        :param numpy.ndarray weighted_sum: nump array of the weighted sums for each node in a layer
        :return numpy.ndarray activations: numpy array of the activation of the nodes in the layer
        """
        return np.maximum(0, weighted_sum)

    @staticmethod
    def relu_prime(activations):
        """
        Derivative of the relu activation function. Vectorized to be single line
        :param numpy.ndarray activations: nump array of the activations for each node in a layer
        :return numpy.ndarray derivatives: numpy array of the activation derivatives of the nodes in the layer
        """
        return (activations > 0) * 1

    @staticmethod
    def softmax(weighted_sum):
        """
        Activation using softmax function
        :param numpy.ndarray weighted_sum: nump array of the weighted sums for each node in a layer
        :return numpy.ndarray activations: numpy array of the activation of the nodes in the layer
        """
        e = np.exp(weighted_sum - np.max(weighted_sum))
        return e / np.sum(e)

    @staticmethod
    def softmax_prime(activations):
        """
        Derivative of the softmax activation function. Uses Jacobian matrix as it was faster than the alternative
        :param numpy.ndarray activations: nump array of the activations for each node in a layer
        :return numpy.ndarray derivatives: numpy array of the activation derivatives of the nodes in the layer. Has
        shape of (amount of nodes, amount of nodes)
        """
        activations_vector = activations.reshape((-1, 1))
        return np.diagflat(activations) - np.dot(activations_vector, activations_vector.T)

    @staticmethod
    def softmax_prime_a(activations):
        """
       Alternative way to find the Derivative of the softmax activation function. This was more common online but was
       slower for me.
       :param numpy.ndarray activations: nump array of the activations for each node in a layer
       :return numpy.ndarray derivatives: numpy array of the activation derivatives of the nodes in the layer. Has
       shape of (amount of nodes, amount of nodes)
       """
        k = len(activations)
        gradient = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    gradient[i, j] = activations[i] * (1 - activations[i])
                else:
                    gradient[i, j] = -activations[i] * activations[j]
        return gradient


if __name__ == '__main__':
    s = np.array([-1, 0, 3, 5])
    print(Activation.softmax(s))
    print(Activation.sigmoid(s))
    print(Activation.sigmoid_prime(Activation.sigmoid(s)))
    print(Activation.tanh(s))
    print(Activation.tanh_prime(Activation.tanh(s)))
    print(Activation.relu(s))
    print(Activation.relu_prime(Activation.relu(s)))

    d_a = Activation.softmax_prime(np.array([0.19091352, 0.20353145, 0.21698333, 0.23132428, 0.15724743]))
    from loss import Loss
    d_l = Loss.cross_entropy_prime(np.array([0.2698, 0.3223, 0.4078, 0.3223, 0.3223]), np.array([0, 0, 1, 0, 0]))
    print(np.dot(d_a, d_l))
    print(d_a)
    print(Activation.softmax_prime_a(np.array([0.19091352, 0.20353145, 0.21698333, 0.23132428, 0.15724743])))
