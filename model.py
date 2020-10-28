from loss import Loss
from activation import  Activation
import numpy as np


class Model:
    # TODO add validation set and early stopping to training
    # TODO have a look at dropout and other regularizing techniques
    # TODO add error handling for mismatching layer input/output
    # ToDo add ability to save and load models

    losses = {
        "cross_entropy": Loss.cross_entropy,
        "binary_cross_entropy": Loss.binary_cross_entropy
    }
    losses_prime = {
        "cross_entropy": Loss.cross_entropy_prime,
        "binary_cross_entropy": Loss.binary_cross_entropy_prime
    }

    activations = {
        "sigmoid": Activation.sigmoid,
        "tanh": Activation.tanh,
        "relu": Activation.relu,
        "softmax": Activation.softmax
    }
    activations_prime = {
        "sigmoid": Activation.sigmoid_prime,
        "tanh": Activation.tanh_prime,
        "relu": Activation.relu_prime,
        "softmax": Activation.softmax_prime
    }

    def __init__(self, loss="cross_entropy"):
        self.layers = []
        self.loss = loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            if type(layer) == "dense.Dense":
                input = input.flatten()
            input = layer.forward(input)
        return input

    def backward(self, gradient, step=0.001):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, step)
        return gradient

    def train(self, data, labels, epochs=3, step=0.001):
        for epoch in range(epochs):
            print("Epoch", epoch + 1)
            order = np.random.permutation(len(data))
            data, labels = data[order], labels[order]

            total_loss = 0
            num_correct = 0
            for i, (image, label) in enumerate(zip(data, labels)):
                out = self.forward(image / 255)
                total_loss += self.calculate_loss(out, label)
                num_correct += 1 if np.argmax(out) == label else 0

                if i == 0 and epoch == 0:
                    print("Step:", i, "Loss:", total_loss)
                elif i % 100 == 0 and i != 0:
                    print("Step:", i, "Past 100 steps: Average loss:", total_loss/100, "Accuracy:", num_correct, "%")
                    total_loss = 0
                    num_correct = 0

                gradient = self.calculate_loss_prime(out, label)
                gradient = self.backward(gradient, step)

    def test(self, data, labels):
        total_loss = 0
        num_correct = 0
        for i, (image, label) in enumerate(zip(data, labels)):
            if i % 100 == 0:
                print(i)
            out = self.forward(image / 255)
            total_loss += self.calculate_loss(out, label)
            num_correct += 1 if np.argmax(out) == label else 0

        print('Test Loss:', total_loss/i)
        print('Test Accuracy:', num_correct/i)

    def calculate_loss(self, out, label):
        return Model.losses[self.loss](out, label)

    def calculate_loss_prime(self, out, target):
        return Model.losses_prime[self.loss](out, target)


if __name__ == '__main__':
    import mnist
    #from keras.datasets import fashion_mnist
    from conv import Conv, ScipyConv, DepthWiseConv
    from pool import Pool
    from dense import Dense


    train_images = mnist.train_images()[:1001]
    train_labels = mnist.train_labels()[:1001]
    print(len(train_images))



    model = Model(loss="cross_entropy")
    model.add_layer(DepthWiseConv(1, 8))
    model.add_layer(Pool())
    model.add_layer(Dense(1352, 10))

    model.train(train_images, train_labels, epochs=3, step=0.001)

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    model.test(test_images, test_labels)
