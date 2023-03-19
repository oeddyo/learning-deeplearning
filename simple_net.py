import numpy as np

from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error
import os
import sys

sys.path.append(os.pardir)


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = SimpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)

print(p)
t = np.array([0, 0, 1])

def f(W):
    return net.loss(x, t)

dw = numerical_gradient(f, net.W)

print(dw)