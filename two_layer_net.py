import numpy as np
from dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error, sigmoid, sigmoid_grad

import os
import sys

sys.path.append(os.pardir)

np.random.seed(42)

class TwoLayerNet:
    def __init__(self):
        v_size = 784
        hidden_size = 50
        output_size = 10

        self.grad = {}

        # 784 * (784, 50) * (50, 10)
        self.params = {"w1": np.random.randn(v_size, hidden_size), "b1": np.zeros(hidden_size),
                       "w2": np.random.randn(hidden_size, output_size), "b2": np.zeros(output_size)}

    def predict(self, x):
        a1 = x @ self.params['w1'] + self.params['b1']
        z1 = sigmoid(a1)

        a2 = z1 @ self.params['w2'] + self.params['b2']
        z2 = softmax(a2)

        return z2

    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['w1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


(x_train, t_train), (x_test, t_test) =  load_mnist(normalize=True, one_hot_label=True)


train_loss_list = []
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 1000
learning_rate = 0.01
network = TwoLayerNet()

iter_per_epoch = max(train_size / batch_size, 1)

print("I will print every ", iter_per_epoch, "steps")

losses = []
accuracies = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # network.step(x_batch, t_batch)

    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) # 高速版!
    for k in ['w1', 'w2', 'b1', 'b2']:
        network.params[k] -= learning_rate * grad[k]

    cur_loss = network.loss(x_batch, t_batch)
    acc = network.accuracy(x_train, t_train)
    if i % 10 == 0:
        print("iter = ", i)
        print('cur loss = ', cur_loss)
        print("current acc = ", acc)
        losses.append(cur_loss)
        accuracies.append(acc)

print("losses ", losses)
print("acc ", accuracies)

