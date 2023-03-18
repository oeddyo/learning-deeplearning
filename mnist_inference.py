import os
import sys
import pickle
import numpy as np
from dataset.mnist import load_mnist
from scipy.special import expit

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def sigmoid(x):
    return expit(x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


x, t = get_data()

network = init_network()

print(len(x), len(t))

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    xb = x[i:i+batch_size]
    y = predict(network, xb)
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])


print("Accuracy = ", accuracy_cnt / len(x))
