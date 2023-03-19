import os
import sys
import numpy as np
from dataset.mnist import load_mnist
from scipy.stats import entropy

sys.path.append(os.pardir)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)


print("ground truth res ", entropy( [1, 0, 0], [0.1, 0.2, 0.7], ))

print(cross_entropy_error(np.array([0.1, 0.2, 0.7]), np.array([1, 0, 0] )))
