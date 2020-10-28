import numpy as np
import math


class Loss:
    # ToDo add validation and error checking for certain loss and activation types
    # ToDo currently only using one-hot encoding, add more
    # ToDo add more losses
    @staticmethod
    def cross_entropy(predictions, target):
        # - sum target[class] log(predicted[class])
        return -np.log(max(predictions[target], 1e-5))

    @staticmethod
    def cross_entropy_prime(predictions, label):
        targets = np.zeros(len(predictions))
        targets[label] = 1
        # - correct[class] / prediction
        return - targets / (predictions + 1e-5)

    @staticmethod
    def binary_cross_entropy(predictions, target):
        # - sum target[class] log(predicted[class])
        return -(target * np.log(predictions[0] + 1e-15) + (1 - target) * np.log(1 - predictions[0] + 1e-15))

    @staticmethod
    def binary_cross_entropy_prime(predictions, label):
        # - correct[class] / prediction
        return - label / predictions + (1 - label) / (1 - predictions)


if __name__ == '__main__':
    print(Loss.cross_entropy(np.array([0.2698, 0.3223, 0.4078]), np.array([1, 0, 0])))
    print(Loss.cross_entropy_prime(np.array([0.2698, 0.3223, 0.4078, 0.3223]), np.array([0, 0, 1, 0])))
    y = [0.09262152, 0.10963767, 0.10006956, 0.09434994, 0.08596,    0.10325457,
         0.09381788, 0.10780763, 0.11752731, 0.09495394]
    t = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    print(Loss.cross_entropy(np.array(y), 6))
